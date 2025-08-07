import asyncio
import boto3
import glob
import io
import json
import logging
import os
import regex
from datetime import datetime
import requests
import hashlib
import calendar
from dateutil.relativedelta import relativedelta

import aiohttp
import PIL.Image as Image
import uvicorn
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, File, UploadFile
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from groq import AsyncGroq
from starlette.status import HTTP_401_UNAUTHORIZED
from typing import Any, Dict, List, Optional, TypedDict
import uuid
import random

from langgraph.graph import StateGraph, START, END
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import googlemaps

from schema import (
    AssistantRequest,
    Location,
    MessageRequest,
    NearbyPlacesRequest,
    PlaceName,
    TravelPlanRequest,
    GenerateMessageRequest,
    ExploreRequest
)

from vapi_setup import create_assistant, get_assistants, update_assistant

load_dotenv()
gmaps_api_key_v2 = os.getenv("GMAPS_API_KEY_V2")
gmaps_api_key_legacy = os.getenv("GMAPS_API_KEY_LEGACY")
GROQAI_API_KEY = os.getenv("GROQAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
groq_client = AsyncGroq(api_key=GROQAI_API_KEY)

aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")

app = FastAPI()
security = HTTPBasic()
current_directory = os.path.dirname(os.path.abspath(__file__))
today_date = datetime.now().strftime("%Y%m%d")
log_file_path = os.path.join(current_directory, f'logs/api_{today_date}.log')
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

llm = ChatGroq(model="llama3-70b-8192", temperature=0, groq_api_key=GROQAI_API_KEY)
gmaps = googlemaps.Client(key=gmaps_api_key_legacy)


# -------------------------------
# Step 1: Define Workflow State
# -------------------------------
class WorkflowState(TypedDict):
    query: str
    places_json: List[Dict]
    enriched_data: List[Dict]
    response: str
    message_history: List[Dict[str, str]]
    user_location: str


# AI Agent 1: Travel Assistant
def travel_assistant(state: WorkflowState):
    """Determines if the query needs place information."""
    logger.info("[Travel Assistant] Processing query...")
    query = state["query"]
    message_history = state["message_history"]
    history_str = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in message_history]
    )
    user_location = state["user_location"]

    # Use LLM to decide whether to redirect
    prompt = PromptTemplate(
        template="""
        You are Trai, a friendly and knowledgeable tourist guide bot and travel assistant AI. 
        Your task is to assist users with their travel-related queries based on the following guidelines:

        - Keep all responses under 60 words unless more detail is explicitly requested.
        - Always consider the context of the user's conversation history to provide relevant and context-aware responses.
        - If the user greets (e.g., "hello", "hi"), introduce yourself and respond warmly with a greeting such as "Hi there! I'm TRAI, your friendly tourist guide bot. How can I help you today?".
        - If this is the first message then only greets the user.
        - If the user asks a travel-related query, provide an informative and helpful response based on the context and query.
        - If the query requires recommending specific places, respond strictly with 'redirect' so another AI agent can handle place recommendations.
        - If the user's location is needed to respond to the query but is not available in the context, politely ask for clarification. For example: "Could you let me know the location you're asking about? This will help me assist you better."
        - Use the provided context (history ) as much as possible.
        - Minimize clarification questions. If critical information is missing, make reasonable assumptions.
        - Be concise, clear, and avoid repeating questions ,instead provide some suggestions.
        - If the user asks about transportation options or how to get somewhere or transport related queries, always include a reminder to verify details from the official transportation website or app for the most up-to-date information.

        Ensure your responses are:
        - Friendly and conversational.
        - Concise and easy to understand.
        - Strictly following the rule to respond with 'redirect' for place recommendations.

        Context:
        - User's conversation history: {history_str}
        - User's location: {user_location}

        User query: {query}
        """,
        input_variables=["query", "history_str", "user_location"],
    )
    result = llm.invoke(prompt.format(query=query, history_str=history_str, user_location=user_location))
    message_history.append({"role": "assistant", "content": result.content})
    state["message_history"] = message_history
    response = {
        "description": result.content,
    }

    if "redirect" in result.content.lower():
        # If redirection is needed
        return {"next_node": "fetch_places"}
    else:
        # General response
        return {"response": response, "places_json": [], "enriched_data": [], "next_node": "END"}


# AI Agent 2: Place Information Provider
def place_provider(state: WorkflowState):
    """Generates JSON-formatted place information based on the query."""
    logger.info("[Place Information Provider] Fetching place details...")
    query = state["query"]
    message_history = state["message_history"]
    history_str = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in message_history]
    )
    user_location = state["user_location"]

    prompt = PromptTemplate(
        template="""
        You are an intelligent AI assistant specializing in providing travel-related, JSON-formatted information about places. 
        Your task is to respond to user queries with accurate and contextual information based on the following:

        - The user's query
        - The user's conversation history (to infer context if the current query is unclear or related to previous topics)
        - The user's location (if provided; otherwise, focus on the query and inferred context)
        - Be concise, clear, and avoid repeating questions ,instead provide some suggestions.
        - Only suggest places within 50km of the user's location
        - If fewer than 5 places are found within 50km, include all available places

        For the given user query, generate a response in proper JSON format (no backticks) that includes:

        - An area description summarizing the general characteristics or significance of the area related to the query or inferred from context.
        - A list of places within 50km radius, with each place containing the following details:
          - Name
          - Detailed Description
          - Address
          - Distance (include distance from user's location)
          - Opening_hours (if applicable, provide operating hours in format "HH:MM AM/PM - HH:MM AM/PM" or "Closed")

        Ensure the response is strictly in proper JSON format.

        Example:
        {{
        "area_description": "New York is a vibrant city known for its cultural diversity, iconic landmarks, and bustling atmosphere.",
        "data": [
                {{"name": "Central Park Cafe", "description": "A cozy cafe known for its artisanal coffee and pastries.", "address": "New York, NY","distance": "2.5 km","opening_hours": "7:00 AM - 8:00 PM"}},
                {{"name": "Statue of Liberty", "description": "A historic monument symbolizing freedom, located on Liberty Island.", "address": "New York Harbor","distance": "3.8 km","opening_hours": "8:30 AM - 4:30 PM"}}

            ]
        }}

        Guidelines:
        - If the query is location-specific, use the provided location and query details to generate the response.
        - If the user's location is not provided, base the response solely on the query and context from conversation history.
        - If the query is vague but the conversation history provides clues about the user's interests or past queries, incorporate those insights into your response.
        - Always include opening_hours field, use "24/7" for places that are always open, or "Varies" if hours are uncertain incase of buissness.
        - Only suggest places within 50km of the user's location
        - Sort places by distance from nearest to farthest

        Context:
        - User's conversation history: {history_str}
        - User's location: {user_location}

        User query: {query}
        """,
        input_variables=["query", "history_str", "user_location"],
    )
    result = llm.invoke(prompt.format(query=query, history_str=history_str, user_location=user_location))
    message_history.append({"role": "assistant", "content": result.content})
    state["message_history"] = message_history
    # logger.info("place_provider AI response: %s", result.content)

    places_json = json.loads(result.content)

    return {"places_json": places_json, "enriched_data": [], "response": ""}


# AI Agent 3: Place Enrichment Agent
def place_enricher(state: WorkflowState):
    """Enriches place information using Google Maps API, including image URLs."""
    logger.info("[Place Enrichment Agent] Fetching additional place details from Google Maps...")
    places = state["places_json"]["data"]
    area_description = state["places_json"]["area_description"]
    places_data = []
    enriched_data = {}

    for place in places:

        try:
            # Include additional context in the search query
            search_query = f"{place['name']}, {place['address']}"

            # Fetch place details with the 'photos' field
            result = gmaps.find_place(
                input=search_query,
                input_type="textquery",
                fields=["formatted_address", "geometry", "name", "photos", "types", "place_id"]
            )

            # Extract the first candidate
            candidate = result["candidates"][0]

            # Get the photo_reference (if available)
            photo_reference = None
            if "photos" in candidate and candidate["photos"]:
                photo_reference = candidate["photos"][0]["photo_reference"]
                photo_url = (
                    f"https://maps.googleapis.com/maps/api/place/photo"
                    f"?maxwidth=400&photo_reference={photo_reference}&key={gmaps_api_key_legacy}"
                )
            else:
                photo_url = "Image not available"

            # Get detailed place information including opening hours and reviews
            place_details_result = gmaps.place(
                place_id=candidate["place_id"],
                fields=[
                    "website",
                    "formatted_phone_number",
                    "international_phone_number",
                    "rating",
                    "user_ratings_total"
                ]
            )

            opening_hours = place.get("opening_hours", "Varies")

            # Append enriched place information
            enriched_place = {
                "name": candidate["name"],
                "address": candidate["formatted_address"],
                "distance": place["distance"],
                "location": candidate["geometry"]["location"],
                "description": place["description"],
                "photo_url": photo_url,
                "place_type": candidate["types"],
                "website": place_details_result.get("result", {}).get("website", ""),
                "contact": place_details_result.get("result", {}).get("international_phone_number", ""),
                "opening_hours": opening_hours,
                "rating": place_details_result.get("result", {}).get("rating"),
                "total_ratings": place_details_result.get("result", {}).get("user_ratings_total"),
            }
            places_data.append(enriched_place)

            enriched_data = {
                "places_data": places_data,
                "area_description": area_description,
            }
        except Exception as e:
            print(f"Error fetching data for {place['name']}: {str(e)}")

    return {"enriched_data": enriched_data, "response": ""}


def explore_place_enricher(places):
    """Enriches place information using Google Maps API, including image URLs."""
    logger.info("[Place Enrichment Agent] Fetching additional place details from Google Maps...")
    places_data = []
    enriched_data = {}
    for place in places:
        try:
            # Include additional context in the search query
            search_query = f"{place['name']}, {place['address']}"

            # Fetch place details with the 'photos' field
            result = gmaps.find_place(
                input=search_query,
                input_type="textquery",
                fields=["formatted_address", "geometry", "name", "photos", "types", "place_id"]
            )
            # Extract the first candidate
            candidate = result["candidates"][0]

            # Get the photo_reference (if available)
            photo_reference = None
            if "photos" in candidate and candidate["photos"]:
                photo_reference = candidate["photos"][0]["photo_reference"]
                photo_url = (
                    f"https://maps.googleapis.com/maps/api/place/photo"
                    f"?maxwidth=400&photo_reference={photo_reference}&key={gmaps_api_key_legacy}"
                )
            else:
                photo_url = "Image not available"

            place_details_result = gmaps.place(
                place_id=candidate["place_id"],
                fields=["website", "formatted_phone_number", "international_phone_number",
                        "rating",
                        "user_ratings_total"]

            )
            # Append enriched place information
            enriched_place = {
                "name": candidate["name"],
                "address": candidate["formatted_address"],
                "location": candidate["geometry"]["location"],
                "description": place.get("description", None),
                "photo_url": photo_url,
                "place_type": candidate["types"],
                "website": place_details_result.get("result").get("website", ""),
                "contact": place_details_result.get("result").get("international_phone_number", ""),
                "rating": place_details_result.get("result").get("rating", ""),
                "total_ratings": place_details_result.get("result").get("user_ratings_total", "")
            }
            places_data.append(enriched_place)

            enriched_data = {
                "places_data": places_data,
            }
        except Exception as e:
            print(f"Error fetching data for {place['name']}: {str(e)}")

    return {"enriched_data": enriched_data, "response": ""}


# Final Response Generator
def generate_final_response(state: WorkflowState):
    """Generates the final response combining enriched data."""
    logger.info("Generating the final response...")
    enriched_data = state["enriched_data"]["places_data"]
    area_description = state["enriched_data"]["area_description"]

    places = []

    for place in enriched_data:
        name_query = place.get("name", "").replace(" ", "+")
        address_query = place.get("address", "").replace(" ", "+")
        mapslink = f"https://www.google.com/maps/search/?api=1&query={name_query}+{address_query}" if name_query and address_query else ""

        # Website validation logic
        website = place.get("website", "")
        if website:
            try:
                # Simple HEAD request with a short timeout
                response = requests.head(website, timeout=3, allow_redirects=True)
                if response.status_code >= 400:  # If website returns error
                    website = mapslink
            except:  # If request fails for any reason
                website = mapslink
        else:
            website = mapslink

        places.append({
            "name": place.get("name", ""),
            "image": place.get("photo_url", ""),
            "website": website,
            "mapslink": mapslink,
            "description": place.get("description", ""),
            "distance": place.get("distance", ""),
            "contact": place.get("contact", ""),
            "opening_hours": place.get("opening_hours", []),
            "rating": place.get("rating", 0),
            "total_ratings": place.get("total_ratings", 0),
        })

    response = {
        "description": area_description,
        "places": places
    }
    return {"response": response}


def explore_final_response(enriched_data):
    """Generates the final response combining enriched data."""
    logger.info("Generating the final response...")
    places = []
    for place in enriched_data:
        name_query = place.get("name", "").replace(" ", "+")
        address_query = place.get("address", "").replace(" ", "+")
        mapslink = f"https://www.google.com/maps/search/?api=1&query={name_query}+{address_query}" if name_query and address_query else ""
        # Website validation logic
        website = place.get("website", "")
        if website:
            try:
                # Simple HEAD request with a short timeout
                response = requests.head(website, timeout=3, allow_redirects=True)
                if response.status_code >= 400:  # If website returns error
                    website = mapslink
            except:  # If request fails for any reason
                website = mapslink
        else:
            website = mapslink
        places.append({
            "name": place.get("name", ""),
            "image": place.get("photo_url", ""),
            "website": website,
            "mapslink": mapslink,
            "description": place.get("description", ""),
            "contact": place.get("contact", ""),
            "rating": place.get("rating", 0),
            "total_ratings": place.get("total_ratings", 0),
        })
    response = {
        "places": places
    }
    return {"response": response}


def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    if not (credentials.username == "admin" and credentials.password == "cJTEBNnn7XoxBbfrr8QD"):
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


async def reverse_geocode(location):
    # logger.info("location", location)
    geocode_endpoint = "https://maps.googleapis.com/maps/api/geocode/json"
    response = requests.get(geocode_endpoint,
                            params={
                                "latlng": location,
                                "result_type": "locality",
                                "key": gmaps_api_key_legacy
                            }
                            )
    geocoded_location = response.json()
    # print("geocoded_location", geocoded_location)
    # logger.info("geocoded_location", geocoded_location)

    return geocoded_location["results"][0]["formatted_address"]


async def extract_routes(location, destination, session):
    try:
        routes_url = "https://routes.googleapis.com/directions/v2:computeRoutes"
        fields = [
            # "routes.routeToken",
            "routes.legs.distanceMeters",
            "routes.legs.duration", "routes.legs.staticDuration",
            "routes.legs.steps.distanceMeters", "routes.legs.steps.staticDuration",
            "routes.legs.steps.navigationInstruction", "routes.legs.steps.travelMode"
        ]
        fields = ','.join(fields)
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": gmaps_api_key_v2,
            "X-Goog-FieldMask": fields,
        }
        payload = {
            "origin": {
                "location": {
                    "latLng": {
                        "latitude": float(location.split(",")[0]),
                        "longitude": float(location.split(",")[1].strip())
                    }
                }
            },
            "destination": {
                "location": {
                    "latLng": {
                        "latitude": float(destination.split(",")[0]),
                        "longitude": float(destination.split(",")[1].strip())
                    }
                }
            },
            "languageCode": "en-US",
            "routingPreference": "TRAFFIC_AWARE"
        }
        async with session.post(routes_url, headers=headers, data=json.dumps(payload)) as response:
            routes = await response.json()

        return routes

    except Exception:
        return None


async def create_presigned_url(bucket_name, object_name, expiration=3600):
    """Generate a presigned URL to share an S3 object

    :param bucket_name: string
    :param object_name: string
    :param expiration: Time in seconds for the presigned URL to remain valid
    :return: Presigned URL as string. If error, returns None.
    """
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region,
    )
    try:
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)
    except NoCredentialsError:
        print("Credentials not available")
        return None

    return response


async def save_photo_to_bucket(photo_reference, max_width):
    try:
        async with aiohttp.ClientSession(trust_env=True) as session:
            place_photo_url = 'https://maps.googleapis.com/maps/api/place/photo'
            params = {
                'maxwidth': max_width,
                'photo_reference': photo_reference,
                'key': gmaps_api_key_legacy
            }
            async with session.get(place_photo_url, params=params) as response:
                if response.status != 200:
                    raise HTTPException(status_code=response.status, detail="Photo fetch failed")

                # Create a shorter filename using MD5 hash
                hash_object = hashlib.md5(photo_reference.encode())
                short_name = hash_object.hexdigest()[:12]  # First 12 chars
                image = Image.open(io.BytesIO(await response.read()))
                image.save(f"{short_name}.jpeg")
                s3 = boto3.client(
                    "s3",
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=aws_region,
                )

                bucket_name = 'placepictures'
                s3_key = f"{short_name}.jpeg"

                with open(f"{short_name}.jpeg", 'rb') as data:
                    s3.upload_fileobj(data, bucket_name, s3_key)

                photo_url = f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{s3_key}"
                # Clean up local jpeg files
                jpeg_files = glob.glob(os.path.join("./", '*.jpeg'))
                for jpeg in jpeg_files:
                    os.remove(jpeg)

                return bucket_name, s3_key, photo_url
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving photo: {str(e)}")


async def process_entry(entry, session, location):
    try:
        location_lat, location_lng = (
            entry["location"]["latitude"],
            entry["location"]["longitude"],
        )

        routes = await extract_routes(location, f"{location_lat},{location_lng}", session)
        photo_list = []
        for photo in entry["photos"][:1]:
            photo_reference = photo["name"].split('photos/')[1]
            photo_max_width = photo["widthPx"]
            bucket_name, object_name, photo_url = await save_photo_to_bucket(photo_reference, photo_max_width)
            # photo_link = await create_presigned_url(bucket_name, object_name)
            photo_list.append(photo_url)

        entry["photos"] = photo_list[:2]
        # entry["routes"] = routes["routes"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing entry: {str(e)}")

    return entry


async def extract_info_for_place(
        place_name: str,
        session: aiohttp.ClientSession,
        destination_place: Optional[str] = None):
    try:
        # First get place details from Google Places API
        search_url = 'https://maps.googleapis.com/maps/api/place/findplacefromtext/json'
        place_input = f"{place_name}, {destination_place}" if destination_place else place_name
        params = {
            'input': place_input,
            'inputtype': 'textquery',
            'fields': 'place_id,formatted_address,name,geometry,types',  # Removed photos field
            'key': gmaps_api_key_legacy
        }

        async with session.get(search_url, params=params) as response:
            data = await response.json()

            if data.get('status') == 'OK' and data.get('candidates'):
                candidate = data['candidates'][0]

                # Get photo using SERP API
                serp_url = "https://serpapi.com/search"
                serp_params = {
                    "q": f"{place_input} landmark tourist attraction",
                    "tbm": "isch",
                    "api_key": SERPAPI_API_KEY,
                    "num": 1  # Get only one image
                }

                async with session.get(serp_url, params=serp_params) as serp_response:
                    serp_data = await serp_response.json()
                    photo_link = None
                    if serp_data.get("images_results") and len(serp_data["images_results"]) > 0:
                        photo_link = serp_data["images_results"][0].get("original")

                # Safely extract geometry information
                geometry = candidate.get('geometry', {})
                location = geometry.get('location', {})
                latitude = location.get('lat')
                longitude = location.get('lng')

                return {
                    'place_id': candidate.get('place_id'),
                    'name': candidate.get('name'),
                    'latitude': latitude,
                    'longitude': longitude,
                    'photos': photo_link,
                    'place_types': candidate.get('types', []),  # Default to empty list if no types
                    'status': data['status']
                }
            else:
                return {'status': 'Error', 'response': data}

    except Exception as e:
        logging.error(f"Error extracting info for place: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error extracting info for place: {str(e)}"
        )


async def enrich_place_info(
        place_info: Dict[str, Any],
        session: aiohttp.ClientSession,
        place_key: str = "place",
        destination_place: Optional[str] = None,
) -> None:
    try:
        place_name = place_info.get(place_key)
        if place_name:
            place_data = await extract_info_for_place(place_name, session, destination_place)
            place_info.update(
                {
                    "latitude": place_data.get("latitude"),
                    "longitude": place_data.get("longitude"),
                    "photos": place_data.get("photos"),
                    "place_type": place_data.get("place_types", [None])[0],
                }
            )
        else:
            place_info.update(
                {
                    "latitude": None,
                    "longitude": None,
                    "photos": None,
                    "place_type": None,
                }
            )
    except Exception as e:
        logger.error(f"Error enriching place info for {place_name}: {str(e)}")


async def convert_preferences_to_place_types(preference):
    supported_place_types = [
        "accounting", "airport", "amusement_park", "aquarium", "art_gallery", "atm",
        "bakery", "bank", "bar", "beauty_salon", "bicycle_store", "book_store",
        "bowling_alley", "bus_station", "cafe", "campground", "car_dealer",
        "car_rental", "car_repair", "car_wash", "casino", "cemetery", "church",
        "city_hall", "clothing_store", "convenience_store", "courthouse", "dentist",
        "department_store", "doctor", "drugstore", "electrician", "electronics_store",
        "embassy", "fire_station", "florist", "funeral_home", "furniture_store",
        "gas_station", "gym", "hair_care", "hardware_store", "hindu_temple",
        "home_goods_store", "hospital", "insurance_agency", "jewelry_store", "laundry",
        "lawyer", "library", "light_rail_station", "liquor_store",
        "local_government_office", "locksmith", "lodging", "meal_delivery",
        "meal_takeaway", "mosque", "movie_rental", "movie_theater", "moving_company",
        "museum", "night_club", "painter", "park", "parking", "pet_store", "pharmacy",
        "physiotherapist", "plumber", "police", "post_office", "primary_school",
        "real_estate_agency", "restaurant", "roofing_contractor", "rv_park", "school",
        "secondary_school", "shoe_store", "shopping_mall", "spa", "stadium", "storage",
        "store", "subway_station", "supermarket", "synagogue", "taxi_stand",
        "tourist_attraction", "train_station", "transit_station", "travel_agency",
        "university", "veterinary_care", "zoo"
    ]

    if preference in supported_place_types:
        return preference

    prompt = (
        f"The following is a list of supported Google Maps place types:\n{supported_place_types}\n\n"
        f"Given the input '{preference}', suggest the closest matching place type "
        "from the list. The response must strictly match one of the entries from the list "
        "without any additional text, formatting, or explanation."
    )
    response = await groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "Return the final type as one word without anything else with it"},
            {"role": "user", "content": prompt}
        ]
    )

    suggestion = response.choices[0].message.content.strip()

    return suggestion


@app.post("/get_nearby_places/")
async def get_nearby_places(
        request: NearbyPlacesRequest
):
    try:
        params = json.loads(request.model_dump_json())

        latitude = params['latitude']
        longitude = params['longitude']
        radius = params['radius']
        preferences = params['preferences']

        location = f"{latitude}, {longitude}"
        nearbysearch_endpoint = (
            "https://places.googleapis.com/v1/places:searchNearby"
        )
        fields = [
            "places.id", "places.displayName.text",
            "places.primaryType", "places.formattedAddress", "places.location",
            "places.rating", "places.editorialSummary",
            # "places.currentOpeningHours.openNow", "places.currentOpeningHours.weekdayDescriptions",
            "places.internationalPhoneNumber", "places.photos.name", "places.photos.widthPx", "places.photos.heightPx",
            "places.priceLevel", "places.websiteUri", "places.reservable",
            "places.allowsDogs", "places.curbsidePickup", "places.delivery",
            "places.dineIn", "places.editorialSummary", "places.fuelOptions",
            "places.goodForChildren", "places.goodForWatchingSports", "places.liveMusic",
            "places.menuForChildren", "places.parkingOptions",
            "places.paymentOptions", "places.outdoorSeating", "places.takeout"

        ]
        fields = ",".join(fields)
        preferences = await asyncio.gather(
            *[convert_preferences_to_place_types(preference) for preference in preferences])

        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': gmaps_api_key_v2,
            'X-Goog-FieldMask': fields
        }
        payload = {
            "includedTypes": preferences,
            "maxResultCount": 5,
            "locationRestriction": {
                "circle": {
                    "center": {
                        "latitude": latitude,
                        "longitude": longitude
                    },
                    "radius": radius
                }
            }
        }

        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.post(
                    nearbysearch_endpoint, headers=headers, data=json.dumps(payload)
            ) as resp:
                if resp.status != 200:
                    raise HTTPException(status_code=resp.status, detail="Error fetching nearby places")

                nearbysearch_response = await resp.json()

                # Ensure the response contains 'places'
                if 'places' not in nearbysearch_response or not nearbysearch_response['places']:
                    raise HTTPException(status_code=404, detail="No places found in the response")

                # Process the results if they exist
                tasks = [
                    process_entry(entry, session, location)
                    for entry in nearbysearch_response['places']
                ]

                results = await asyncio.gather(*tasks)

            return results

    except aiohttp.ClientError as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to Google Places API: {str(e)}")

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON response from API: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/setup_assistant/")
async def setup_assistant(request: AssistantRequest):
    _address = await reverse_geocode(request.address)
    system_prompt = f"""
    You are a tourist guide bot, based on the user's preferences and location, you will give the user recommendations of various (at least 2 or 3)
    points of interests and activities they can visit so they can have the best traveling experience.\n
    Address the user in a casual and friendly way and try to recommend them available activities and gauge their feedback to tailor your recommendation,
    try to make your recommendations as concise as possible while giving them all the information they need about the places 
    and prompt the user for more details or other options if they please\n Context: Address {_address}
    """

    first_message = f"""Hey there! I'm your travel assistant and welcome to {_address}, 
    great to have the pleasure to assist you. To kick things off, could you share what kind of activities or sights you're most interested in? 
    Are you into historical sites, shopping, food experiences, nature, or perhaps something else? 
    Let me know, and I'll tailor some recommendations just for you!"""

    kb_file = 'knowledgebase_file.json'
    with open(kb_file, 'w') as f:
        json.dump(json.loads(request.model_dump_json())["nearby_places"], f)

    assistants = get_assistants()
    if len(assistants) == 0:
        response = create_assistant(system_prompt, first_message, kb_file)
    else:
        response = update_assistant(assistants[0]["id"], system_prompt, first_message, kb_file)

    return response


@app.post("/generate_message/")
async def generate_message(request: GenerateMessageRequest):
    logger.info("prompt from request: %s", request.prompt)
    logger.info("latitude from request: %s", request.latitude)
    logger.info("longitude from request: %s", request.longitude)
    logger.info("email from request: %s", request.email)
    logger.info("session from request: %s", request.session_id)
    logger.info("session from request: %s", request.message_history)

    _address = ""
    if request.latitude and request.longitude:
        address = f"{request.latitude}, {request.longitude}"
        _address = await reverse_geocode(address)

    workflow = StateGraph(WorkflowState)

    # Define Nodes
    workflow.add_node("travel_assistant", travel_assistant)
    workflow.add_node("place_provider", place_provider)
    workflow.add_node("place_enricher", place_enricher)
    workflow.add_node("generate_response", generate_final_response)

    # Define Edges
    workflow.add_edge(START, "travel_assistant")
    workflow.add_conditional_edges(
        "travel_assistant",
        lambda state: state["next_node"],  # Extract the next node from the state
        {"fetch_places": "place_provider", "END": END},
    )
    workflow.add_edge("place_provider", "place_enricher")
    workflow.add_edge("place_enricher", "generate_response")
    workflow.add_edge("generate_response", END)

    # Compile Workflow
    custom_graph = workflow.compile()

    """Executes the multi-agent workflow."""
    result = custom_graph.invoke({
        "query": request.prompt,
        "places_json": [],
        "enriched_data": [],
        "response": "",
        "message_history": [{"role": msg.role, "content": msg.content} for msg in request.message_history],
        "user_location": _address
    })

    response = {
        "response": result["response"],
        "user": request.email,
        "session_id": request.session_id,
        "latitude": request.latitude,
        "longitude": request.longitude,
        "prompt": request.prompt,
        "message_history": request.message_history,
    }

    return response


async def generate_6_random_places():
    """Generate 6 touristic cities based."""
    seasons = ["winter", "spring", "summer", "autumn"]
    season = seasons[datetime.now().month // 3]
    system_prompt = """You are an Tourist guide."""
    user_prompt = f"""Generate a random list of 6 cities based on the current season: {season}.
    The structure of the JSON must be the following:
    {{"cities": [{{"city":"city 1","Description":"description 1"}},{{"city":"city 2","Description":"description 2"}}, ... ,{{"city":"city 6","Description":"description 6"}}]}}
    where city field is the city that is suggested to be visited, it might be a city, town or country. Description is a description of that city for the tourist. list of the 6 cities in JSON format:"""

    response = await groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content


@app.get("/generate-6-random-places/")
async def get_6_random_places():
    try:
        places = await generate_6_random_places()
        # Parse the JSON from the travel plan
        pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')
        places_json = pattern.findall(places)[0]
        return json.loads(places_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def generate_travel_plan(
        n_travel_days: int,
        destination_place: str,
        user_interests: List[str],
        user_company: str,
) -> str:
    """Generate a travel plan using Groq, considering the user's company during the trip."""
    system_prompt = """
    You are an expert travel planner. Your task is to generate a complete, valid JSON travel plan.

    IMPORTANT GUIDELINES:
    1. Ensure your entire response is a valid, complete JSON object
    2. Verify that all JSON syntax is correct - properly formatted with quotes, commas, and braces
    3. Do not include any explanatory text before or after the JSON
    4. Make sure all Places sections for each day are complete and properly nested
    5. All numeric keys in Places must be strings (e.g. "1", "2", etc.)
    6. Do not include comments or explanations inside the JSON
    7. Return only the JSON object as your complete response
    8. Double-check all string values have opening and closing quotes
    9. Ensure there are no trailing commas at the end of lists or objects
    10. All URLs must be properly formatted with closing quotes
    11. Do not use the same key twice in the same object (e.g., don't repeat "link" key)
    """

    user_prompt = f"""
Generate a travel plan of {n_travel_days} days for a person who wants to visit {destination_place}. The person's interests are: {', '.join(user_interests)}. This person is traveling with {user_company}.

For each day, generate exactly 5 places, following this structure for EACH place:
1. Each place must include: place, description, meal, link, duration, and transportation fields
2. Meal field should contain "breakfast", "lunch", "dinner", or "" if not applicable
3. All links must be valid URLs with proper quotation marks
4. Description should be a brief, descriptive text about the place
5. Duration is how long to spend there (e.g., "2 hours")
6. Transportation is how to get to the next place (e.g., "Walk to the nearby restaurant")

The structure of the JSON must be the following (follow this EXACT format):
{{
  "Day 1": {{
    "exclusive_data": "Brief description about the destination",
    "Places": {{
      "1": {{
        "place": "Name of place 1",
        "description": "Description of place 1",
        "meal": "breakfast/lunch/dinner or empty string",
        "link": "URL for place 1",
        "duration": "Time to spend here",
        "transportation": "How to get to place 2"
      }},
      "2": {{
        "place": "Name of place 2",
        "description": "Description of place 2",
        "meal": "breakfast/lunch/dinner or empty string",
        "link": "URL for place 2",
        "duration": "Time to spend here",
        "transportation": "How to get to place 3"
      }},
      "3": {{
        "place": "Name of place 3",
        "description": "Description of place 3",
        "meal": "breakfast/lunch/dinner or empty string",
        "link": "URL for place 3",
        "duration": "Time to spend here",
        "transportation": "How to get to place 4"
      }},
      "4": {{
        "place": "Name of place 4",
        "description": "Description of place 4",
        "meal": "breakfast/lunch/dinner or empty string",
        "link": "URL for place 4",
        "duration": "Time to spend here",
        "transportation": "How to get to place 5"
      }},
      "5": {{
        "place": "Name of place 5",
        "description": "Description of place 5",
        "meal": "breakfast/lunch/dinner or empty string",
        "link": "URL for place 5",
        "duration": "Time to spend here",
        "transportation": "How to get back to hotel"
      }}
    }}
  }},
  "Day 2": {{ ... similar structure ... }},
  "Day 3": {{ ... similar structure ... }}
}}

Provide ONLY a properly formatted JSON object as your complete response. All string values must be enclosed in double quotes and properly escaped where necessary.
"""

    try:
        response = await groq_client.chat.completions.create(
            model="llama3-70b-8192",
            temperature=0.1,  # Lower temperature for more consistent output
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling Groq API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating travel plan: {str(e)}")


@app.post("/generate-travel-plan/")
async def create_travel_plan(request: TravelPlanRequest):
    try:
        travel_plan = await generate_travel_plan(
            request.n_travel_days,
            request.destination_place,
            request.user_interests,
            request.user_company,
        )

        logging.info(f"Travel plan generated: {travel_plan}")

        # Try to clean and repair the JSON directly from the text
        # Start by removing non-JSON content
        travel_plan = travel_plan.strip()
        if travel_plan.startswith('```json'):
            travel_plan = travel_plan[7:]
        if travel_plan.startswith('```'):
            travel_plan = travel_plan[3:]
        if travel_plan.endswith('```'):
            travel_plan = travel_plan[:-3]
        travel_plan = travel_plan.strip()

        # Basic cleanup operations
        # Remove control characters
        travel_plan = ''.join(ch for ch in travel_plan if ch >= ' ')

        # Try to parse directly first
        try:
            travel_plan_dict = json.loads(travel_plan)
            logger.info("Successfully parsed JSON directly")
        except json.JSONDecodeError as json_err:
            # If that fails, do a more thorough repair
            logger.error(f"Initial JSON parsing error: {str(json_err)}")

            # Fall back to manual structure building
            logger.info("Falling back to manual JSON structure building")
            travel_plan_dict = manual_json_structure_builder(travel_plan, request.n_travel_days)

        # Replace empty strings with None
        travel_plan_dict = replace_empty_strings_with_none(travel_plan_dict)

        # Skip enrichment if we're in a test/debug mode
        if os.environ.get("SKIP_ENRICHMENT") == "1":
            return travel_plan_dict

        # Enrich the data with place information
        async with aiohttp.ClientSession(trust_env=True) as session:
            enrichment_tasks = []
            for day_key, day_activities in travel_plan_dict.items():
                # Normalize day_activities keys to ensure consistent structure
                if "exclusive data" in day_activities and "exclusive_data" not in day_activities:
                    day_activities["exclusive_data"] = day_activities.pop("exclusive data")

                for places_key, places in day_activities.items():
                    if isinstance(places, dict):
                        for place_key, place_info in places.items():
                            # Skip if place_info is None or not a dict
                            if not place_info or not isinstance(place_info, dict):
                                continue

                            # Ensure all required fields exist
                            for field in ["place", "description", "meal", "link", "duration", "transportation"]:
                                if field not in place_info:
                                    place_info[field] = "" if field != "place" else "Unknown Place"

                            # Create Google Maps link for each place
                            if place_info.get('place'):
                                place_name = place_info['place']
                                destination = request.destination_place.lower()

                                # Check if destination is already part of the place name
                                if destination not in place_name.lower():
                                    place_query = f"{place_name}, {request.destination_place}"
                                else:
                                    place_query = place_name

                                place_query_encoded = place_query.replace(' ', '+')
                                place_info[
                                    'link'] = f"https://www.google.com/maps/search/?api=1&query={place_query_encoded}"

                            enrichment_tasks.append(
                                enrich_place_info(place_info, session, "place",
                                                  destination_place=request.destination_place)
                            )

            await asyncio.gather(*enrichment_tasks)

        return travel_plan_dict

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error parsing travel plan JSON: {str(e)}"
        )
    except ValueError as e:
        logger.error(f"Value error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except aiohttp.ClientError as e:
        logger.error(f"AIOHTTP client error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error during API request: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )


def manual_json_structure_builder(json_text, n_days):
    """
    Builds a structured JSON object from malformed travel plan text.
    This is a fallback method when normal parsing fails.
    """
    result = {}

    # Extract each day
    for day_num in range(1, n_days + 1):
        day_key = f"Day {day_num}"
        day_pattern = f'"{day_key}"\\s*:\\s*{{([^}}}}]*(?:{{[^}}]*}}[^}}}}]*)*)}}'
        day_match = regex.search(day_pattern, json_text)

        if not day_match:
            # If day not found, create a placeholder
            result[day_key] = {
                "exclusive_data": f"Information about {day_key}",
                "Places": {}
            }
            continue

        day_content = day_match.group(1)
        day_data = {}

        # Extract exclusive data
        exclusive_data_pattern = r'"exclusive_?data"\s*:\s*"([^"]*)"'
        exclusive_match = regex.search(exclusive_data_pattern, day_content)
        if exclusive_match:
            day_data["exclusive_data"] = exclusive_match.group(1)
        else:
            day_data["exclusive_data"] = f"Information about {day_key}"

        # Extract places
        places_data = {}
        for place_num in range(1, 6):  # Assume 5 places per day
            place_pattern = f'"?{place_num}"?\\s*:\\s*{{([^}}]*(?:{{[^}}]*}}[^}}]*)*)}}'
            place_match = regex.search(place_pattern, day_content)

            if not place_match:
                # If place not found, create a placeholder
                places_data[str(place_num)] = {
                    "place": f"Place {place_num}",
                    "description": "Description unavailable",
                    "meal": "",
                    "link": "",
                    "duration": "1 hour",
                    "transportation": "Walk"
                }
                continue

            place_content = place_match.group(1)
            place_data = {}

            # Extract fields from place
            field_patterns = {
                "place": r'"place"\s*:\s*"([^"]*)"',
                "description": r'"description"\s*:\s*"([^"]*)"',
                "meal": r'"meal"\s*:\s*"([^"]*)"',
                "link": r'"link"\s*:\s*"([^"]*)"',
                "duration": r'"duration"\s*:\s*"([^"]*)"',
                "transportation": r'"transportation"\s*:\s*"([^"]*)"'
            }

            for field, pattern in field_patterns.items():
                field_match = regex.search(pattern, place_content)
                if field_match:
                    place_data[field] = field_match.group(1)
                else:
                    # Default values if field not found
                    if field == "place":
                        place_data[field] = f"Place {place_num}"
                    elif field == "description":
                        place_data[field] = "Description unavailable"
                    elif field == "meal":
                        place_data[field] = ""
                    elif field == "link":
                        place_data[field] = ""
                    elif field == "duration":
                        place_data[field] = "1 hour"
                    elif field == "transportation":
                        place_data[field] = "Walk"

            places_data[str(place_num)] = place_data

        day_data["Places"] = places_data
        result[day_key] = day_data

    return result


def replace_empty_strings_with_none(data):
    """
    Recursively traverse the dictionary or list and replace empty strings with None.
    """
    if isinstance(data, dict):
        return {
            key: replace_empty_strings_with_none(value) for key, value in data.items()
        }
    elif isinstance(data, list):
        return [replace_empty_strings_with_none(item) for item in data]
    elif data == "":
        return None
    else:
        return data


async def get_popular_destination():
    # Get current month and calculate next 6 months
    current_date = datetime.now()
    months_range = [(current_date.replace(day=1) + relativedelta(months=i)).month for i in range(5)]
    months_names = [calendar.month_name[m] for m in months_range]

    system_prompt = """You are a luxury travel expert specializing in trending destinations and seasonal travel recommendations."""
    user_prompt = f"""
        Suggest ten exclusive and trending tourist destinations suitable for visits during {', '.join(months_names)}. Your suggestions should:

        1. Include destinations that are:
           - Internationally recognized tourist destinations
           - Premium luxury destinations (e.g., Maldives, Bora Bora, Seychelles)
           - Popular tourist hotspots (e.g., Santorini, Bali, Dubai)
           - From different continents (ensure geographic diversity)
           - Having ideal conditions during any of the mentioned months
           - Known for luxury travel experiences
           - Currently trending on social media and travel blogs
           - Offering exceptional hospitality and amenities

        2. Focus on destinations where at least one of the mentioned months aligns with:
           - Optimal weather conditions
           - Peak season for unique experiences
           - Major cultural events or festivals
           - Best conditions for outdoor activities

        3. Consider destinations known for:
           - Premium accommodations
           - Unique natural beauty
           - Exclusive activities (private tours, yacht trips, etc.)
           - Wellness and spa experiences
           - Gourmet dining

        Format the response as a JSON array with exactly 10 objects, including only these fields:
        [
            {{"city": "Major City Name", "state": "state name", "country": "Country Name"}},
            // ... (total of 10 entries)
        ]
        Return only the JSON output. Focus on destinations where timing during the specified months offers the best possible experience.
        """

    response = await groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content


# Endpoint to get nearby cities based on location data
@app.post("/popular-destination/")
async def popular_destination():
    try:
        # Get initial data
        cities_data = await get_popular_destination()

        # Extract and parse JSON in one step
        json_pattern = regex.compile(r"\{(?:[^{}]|(?R))*\}")
        cities_list = [json.loads(city) for city in json_pattern.findall(cities_data)]

        if not cities_list:
            raise ValueError("No valid JSON data found in the response.")

        # Enrich all cities in parallel
        async with aiohttp.ClientSession(trust_env=True) as session:
            # Create enrichment tasks
            enrichment_tasks = [
                enrich_place_info(city, session, "city", destination_place=city.get("state"))
                for city in cities_list
            ]

            # Run all enrichment tasks concurrently
            await asyncio.gather(*enrichment_tasks)

            # Filter and select complete cities in one pass
            complete_cities = [
                city for city in cities_list
                if all([
                    city.get("latitude") is not None,
                    city.get("longitude") is not None,
                    city.get("photos") is not None,
                    city.get("place_type") is not None,
                    city.get("city"),
                    city.get("country")
                ])
            ]

            # Select final cities
            selected_cities = (
                random.sample(complete_cities, 6)
                if len(complete_cities) >= 6
                else complete_cities
            )

            logger.info(f"Selected {len(selected_cities)} complete cities")
            return selected_cities

    except Exception as e:
        logger.error(f"Error in popular_destination: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the request: {str(e)}"
        )


async def get_place_details(place: str):
    # Prepare the payload based on the provided prompt
    system_prompt = """You are an Tourist guide."""
    user_prompt = f"""
        Your task is to provide a one-line with max 20 words, engaging description of {place}, focusing on the following categories: Beach, Art, Attractions, Adventure, Food, Bars, Nightlife, Shopping, Sports, or Culture. write some intriguing and lesser-known facts for any two chosen categories.

        Output should be formatted as:
        [
            {{"category": "Beach", "text": "Interesting detail about the beach."}},
            {{"category": "Culture", "text": "Cultural insight about the destination."}},
            {{"category": "Art", "text": "Fascinating fact about local art."}},
            {{"category": "Food", "text": "Unique culinary experience in the area."}},
            {{"category": "Nightlife", "text": "Exciting nightlife feature."}}
        ]

        Ensure you select 5 different categories that are most relevant to {place}. If a category isn't applicable (e.g., no beaches), replace it with another category from the list.
        """

    response = await groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content


@app.post("/place-details/")
async def place_details(request: PlaceName):
    try:
        # Get place details from the AI model
        place_data = await get_place_details(request.place)

        # Extract JSON-like objects from the response using regex
        json_pattern = regex.compile(r"\{(?:[^{}]|(?R))*\}")
        place_json_list = json_pattern.findall(place_data)

        # Log the extracted JSON data
        logger.info(f"Extracted Place details JSON: {place_json_list}")

        # Raise an exception if no JSON data is found
        if not place_json_list:
            raise ValueError("No valid JSON data found in the response.")

        # Parse the JSON strings into Python objects
        place_list = [json.loads(place) for place in place_json_list]
        return place_list

    except json.JSONDecodeError:
        # Handle JSON parsing errors
        raise HTTPException(status_code=400, detail="Invalid JSON format in the response.")

    except Exception as e:
        # Handle other exceptions
        logger.error(f"Error processing place details: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the request.")


async def explore_place(place: str, month: str = None):
    system_prompt = "You are a comprehensive City guide with extensive knowledge about local culture, food, history, and attractions."

    weather_prompt = (
        f"Focus on conditions in {month}, including average temperature, precipitation, and weather patterns."
        if month else "Provide year-round overview of climate patterns and seasonal variations.")

    # Simplified JSON structure with reduced word count requirements
    base_prompt = f"""
        Provide detailed information about {place} in the following categories. Each category should have concise but informative details.

        The output must be formatted as a JSON object with the following structure:
        {{
            "weather": {{
                "description": "Briefed weather information including temperature ranges and seasonal patterns"
            }},
            "food": {{
                "description": "Comprehensive overview of local cuisine and food culture. Must be at least 500 words covering regional specialties, cooking styles, dining customs, food markets, seasonal delicacies, street food culture, and local ingredients.",
                "restaurants": [
                    {{
                        "name": "Restaurant Name",
                        "address": "address",
                        "description": "Detailed description covering cuisine style, ambiance, signature dishes, and any unique features"
                    }},
                    // ... more restaurants (total of 10)
                ]
            }},
            "history": {{
                "description": "Detailed description of historical significance"
            }},
            "transportation": {{
                "description": "Detailed description of the city's transportation system. Must be at least 500 words covering the overall network, main transport modes, accessibility, tourist-friendly features, operating hours, and general pricing structure. Include information about connectivity between different areas and tips for visitors.",
                "options": [
                    {{
                        "name": "Transport hub name",
                        "mode": "Type of transport",
                        "address": "Full address",
                        "description": "Details about service, coverage, operating hours, and fares"
                    }},
                    // ... more transport options
                ]
            }},
            "shopping": {{
                "description": "Comprehensive overview of the shopping landscape. IMPORTANT: Your response must contain at least 500 words. Describe: 1) Main shopping districts and areas 2) Types of retail experiences (malls, markets, boutiques) 3) Price ranges and budget considerations 4) Local specialties and unique products 5) Best times to shop 6) Traditional vs modern shopping options 7) Notable shopping streets or zones 8) Special shopping events or seasons.",
                "places": [
                    {{
                        "name": "Shop/Mall name",
                        "address": "Full address",
                        "description": "Detailed description of products, specialties, and shopping experience",
                        "type": "Type of shopping venue"
                    }},
                    // ... more shopping places (total of 3)
                ]
            }},
            "traditions": {{
                "description": "Detailed description of local customs and cultural practices. IMPORTANT: Your response must contain at least 500 words. Include: 1) Major annual festivals and their significance 2) Important ceremonies and rituals 3) Local social etiquette and customs 4) Traditional practices in daily life 5) Seasonal celebrations and their meaning 6) Cultural values and beliefs 7) Historical origins of traditions 8) Modern interpretations of customs.",
                "events": [
                    {{
                        "name": "Festival/Tradition name",
                        "description": "Brief description of the tradition minimum 80 words."
                    }},
                    // ... more traditions (total of 3)
                ]
            }},
            "must_visit": {{
                "description": "Detailed overview of essential attractions and landmarks. Must be at least 500 words covering the city's most significant sites, their historical importance, cultural value, best visiting times, and unique features. Include both famous landmarks and hidden gems that define the destination's character.",
                "places": [
                    {{
                        "name": "Attraction name",
                        "address": "Full address",
                        "description": "Brief description",
                        "type": "Type of attraction"
                    }},
                    // ... more attractions (total of 3)
                ]
            }}
        }}

        Guidelines for each category:
        - Weather: {weather_prompt}
          * brief description in max 100 words.
        - Food: Include 10 specific restaurant recommendations with 
          * Actual restaurant names and addresses
          * Brief description of the cuisine and ambiance
          * Their specialty or signature dish
          * Describe the dining atmosphere and experience
          * Include resturents that are in running state
        - History: 
          * Provide a comprehensive yet concise overview of the destination's history
          * Include key historical periods, major events, and cultural milestones
          * Highlight significant architectural developments and cultural evolution
          * Focus on historical facts that shaped the current identity of the place
          * Include notable historical figures and their contributions
          * Mention any UNESCO World Heritage sites or historically protected areas
          * Discuss the influence of history on modern culture and lifestyle
          * Describe any historical districts or quarters worth visiting
          * Detailed Description in min 300 words, focusing on the most impactful elements

        - Transportation: Include all available transport modes with specific details
          * List all major public transportation modes (metro, bus, train, etc.)
          * Include names and exact addresses of main stations/terminals for each mode
          * Include operating hours for each mode
          * Mention typical fare ranges
          * Note any special passes or cards for tourists
          * Include information about major taxi stands and ride-sharing pickup points
          * Mention main bicycle rental stations if available
          * Detailed Description in min 500 words.
        - Shopping:
          * Include both traditional markets and modern shopping centers
          * Specify price ranges and best times for shopping
          * Highlight local specialties and unique products
          * Provide practical shopping tips for visitors
          * Detailed Description in min 500 words.
        - Traditions:
          * Detail major festivals with specific dates and locations
          * Explain cultural significance and modern interpretations
          * Include practical tips for visitors participating in festivals  
          * Detailed Description in min 500 words.
        - Must Visit:
          * List top 3 attractions with brief descriptions
          * Detailed Description in min 500 words.

        Keep descriptions informative and practical (min 500 words per field except weather field).
        Ensure all recommendations are currently operating establishments.
        Include only transport modes that actually exist in {place}.
        """

    # Add month-specific weather information if provided

    response = await groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": base_prompt}
        ]
    )

    try:
        # Parse and return the JSON response
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error processing place data: {str(e)}")
        return response.choices[0].message.content


@app.post("/explore-place-details/")
async def explore_place_details(request: ExploreRequest):
    logger.info("place from request: %s", request.place_name)
    logger.info("trip start date from request: %s", request.trip_start_date)
    logger.info("trip end date from request: %s", request.trip_end_date)

    try:
        month = None
        # Extract month from trip_start_date only if dates are provided
        if request.trip_start_date and request.trip_end_date:
            month = datetime.strptime(request.trip_start_date, "%Y-%m-%d").strftime("%B")

        place_data = await explore_place(request.place_name, month)
        json_pattern = regex.compile(r"\{(?:[^{}]|(?R))*\}")

        try:
            data = json.loads(json_pattern.findall(place_data)[0])

            # Define sections to enrich
            sections_to_enrich = [
                ("food", "restaurants"),
                ("must_visit", "places"),
                ("transportation", "options"),
                ("shopping", "places"),
                ("traditions", "events")
            ]

            # Enrich all sections
            for section, subsection in sections_to_enrich:
                await enrich_data_section(data, section, subsection, request.place_name)

            # Handle history section separately
            await enrich_data_section(data, "history", place_name=request.place_name)

            return data

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON string: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON format in the response.")

    except ValueError as e:
        if "time data" in str(e):
            raise HTTPException(status_code=400, detail="Invalid date format. Please use YYYY-MM-DD")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Handle other exceptions
        logger.error(f"Error processing place details: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the request.")


async def enrich_data_section(data: dict, section_name: str, subsection_name: str = None, place_name: str = None):
    """Helper function to enrich data sections with place information."""
    if section_name not in data:
        return

    # Early return if subsection is required but missing
    if section_name not in ("history", "traditions") and (
            not subsection_name or subsection_name not in data[section_name]):
        return

    # Initialize common variables
    url = "https://serpapi.com/search"

    # Handle different section types
    if section_name == "history":
        await enrich_history_section(data, section_name, place_name, url)
    elif section_name == "traditions":
        await enrich_traditions_section(data, section_name, subsection_name, url)
    else:
        await enrich_other_section(data, section_name, subsection_name)


async def enrich_history_section(data: dict, section_name: str, place_name: str, url: str):
    """Enrich history section with images and maps link."""
    if "images" not in data[section_name]:
        data[section_name]["images"] = []

    params = {
        "q": f"{place_name} historical landmarks tourist attractions",
        "tbm": "isch",
        "api_key": SERPAPI_API_KEY,
        "num": 2
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            results = await response.json()

    if results.get("images_results"):
        image_urls = [result["original"] for result in results["images_results"][:2]]
        data[section_name]["images"] = image_urls
        name_query = place_name.replace(" ", "+")
        data[section_name]["mapslink"] = f"https://www.google.com/maps/search/?api=1&query={name_query}"


async def enrich_traditions_section(data: dict, section_name: str, subsection_name: str, url: str):
    """Enrich traditions section with images."""
    original_events = data[section_name][subsection_name]

    async with aiohttp.ClientSession() as session:
        tasks = []
        for event in original_events:
            task = fetch_tradition_image(session, event, url)
            tasks.append(task)
        await asyncio.gather(*tasks)


async def fetch_tradition_image(session: aiohttp.ClientSession, event: dict, url: str):
    """Fetch image for a tradition event."""
    try:
        params = {
            "q": f"{event['name']} festival tradition",
            "tbm": "isch",
            "api_key": SERPAPI_API_KEY,
            "num": 1
        }

        async with session.get(url, params=params) as response:
            results = await response.json()

        if results.get("images_results"):
            event["image"] = results["images_results"][0].get("original")
        else:
            event["image"] = None

    except Exception as e:
        logger.error(f"Error fetching image for event {event['name']}: {str(e)}")
        event["image"] = None


async def enrich_other_section(data: dict, section_name: str, subsection_name: str):
    """Enrich other sections with place data."""
    place_data = data[section_name][subsection_name]
    enricher_response = explore_place_enricher(place_data)
    result = explore_final_response(enricher_response['enriched_data']["places_data"])

    # Sort restaurants by rating if this is the food section
    if section_name == "food" and subsection_name == "restaurants":
        places = result["response"]["places"]
        # Sort by rating (descending) and handle None values
        sorted_places = sorted(
            places,
            key=lambda x: (x.get("rating") or 0, x.get("total_ratings") or 0),
            reverse=True
        )
        data[section_name][subsection_name] = sorted_places
    else:
        data[section_name][subsection_name] = result["response"]["places"]


@app.post("/upload-profile-pic/")
async def upload_profile_pic(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = await file.read()

        # Generate a unique filename using UUID
        file_extension = file.filename.split('.')[-1]
        unique_filename = f"profile_pics/{str(uuid.uuid4())}.{file_extension}"

        # Create a temporary file to process the image
        with open(f"temp_{file.filename}", "wb") as temp_file:
            temp_file.write(contents)

        # Open and optimize the image
        with Image.open(f"temp_{file.filename}") as img:
            # Convert to RGB if needed
            # if img.mode in ('RGBA', 'P'):
            #     img.convert('RGB')

            # Resize if too large (optional)
            max_size = (800, 800)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

            # Save optimized image
            img.save(f"temp_optimized_{file.filename}", optimize=True, quality=85)

        # Upload to S3
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region,
        )

        bucket_name = 'placepictures'  # Using the same bucket as other images

        # Upload the optimized file
        with open(f"temp_optimized_{file.filename}", 'rb') as data:
            s3.upload_fileobj(data, bucket_name, unique_filename)

        # Generate the S3 URL
        s3_url = f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{unique_filename}"

        # Clean up temporary files
        import os
        os.remove(f"temp_{file.filename}")
        os.remove(f"temp_optimized_{file.filename}")

        return {"success": True, "profile_pic_url": s3_url}

    except Exception as e:
        logger.error(f"Error uploading profile picture: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading profile picture: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
