from pydantic import BaseModel
from typing import List, Optional

class Location(BaseModel):
    latitude: float
    longitude: float

class DisplayName(BaseModel):
    text: str

class PlaceName(BaseModel):
    place: str

class CurrentOpeningHours(BaseModel):
    openNow: bool
    weekdayDescriptions: Optional[List[str]] = None

class EditorialSummary(BaseModel):
    text: str
    languageCode: str

class NavigationInstruction(BaseModel):
    maneuver: str
    instructions: str

class Step(BaseModel):
    distanceMeters: Optional[int] = None
    staticDuration: Optional[str] = None
    navigationInstruction: NavigationInstruction
    travelMode: Optional[str] = None

class Leg(BaseModel):
    distanceMeters: int
    duration: str
    staticDuration: str
    steps: List[Step]

class Route(BaseModel):
    legs: List[Leg]
    # routeToken: str

class PaymentOptions(BaseModel):
    acceptsCreditCards: Optional[bool] = None
    acceptsDebitCards: Optional[bool] = None
    acceptsCashOnly: Optional[bool] = None

class ParkingOptions(BaseModel):
    freeParkingLot: Optional[bool] = None
    freeStreetParking: Optional[bool] = None

class Place(BaseModel):
    id: str
    internationalPhoneNumber: Optional[str] = None
    formattedAddress: str
    location: Location
    rating: Optional[float] = None
    websiteUri: Optional[str] = None
    priceLevel: Optional[str] = None
    displayName: DisplayName
    takeout: Optional[bool] = None
    delivery: Optional[bool] = None
    dineIn: Optional[bool] = None
    reservable: Optional[bool] = None
    # currentOpeningHours: Optional[CurrentOpeningHours] = None
    primaryType: Optional[str] = None
    editorialSummary: Optional[EditorialSummary] = None
    photos: Optional[List[str]] = None
    outdoorSeating: Optional[bool] = None
    liveMusic: Optional[bool] = None
    menuForChildren: Optional[bool] = None
    goodForChildren: Optional[bool] = None
    paymentOptions: Optional[PaymentOptions] = None
    parkingOptions: Optional[ParkingOptions] = None
    # routes: Optional[List[Route]] = None

class Message(BaseModel):
    role: str
    content: dict

class NearbyPlacesRequest(BaseModel):
    latitude:float
    longitude: float
    radius: float
    preferences: Optional[list[str]]

class MessageRequest(BaseModel):
    message_history: List[Message]
    prompt: str
    nearby_places: List[Place]
    address: str
    email: str
    session_id: str

class GenerateMessageRequest(BaseModel):
    prompt: str
    latitude: Optional[float]= None
    longitude: Optional[float]= None
    email: str
    session_id: str
    message_history: List[Message]

class AssistantRequest(BaseModel):
    nearby_places: List[Place]
    address: str

class TravelPlanRequest(BaseModel):
    n_travel_days: int
    destination_place: str
    user_interests: list[str]
    user_company: str

class ExploreRequest(BaseModel):
    email: Optional[str] = None
    place_name: str
    trip_start_date: Optional[str] = None
    trip_end_date: Optional[str] = None


