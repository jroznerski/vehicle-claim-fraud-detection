from typing import Optional
from fastapi import FastAPI
import joblib
from pydantic import BaseModel, ConfigDict, Field
import pandas as pd

app = FastAPI()
model = joblib.load("fraud_model_pipeline.pkl")


class ClaimData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # numeric / ordinal
    Month: int
    WeekOfMonth: int
    MonthClaimed: int
    WeekOfMonthClaimed: int
    Age: Optional[float] = None
    VehiclePrice: int
    RepNumber: int
    Deductible: Optional[float] = None
    DriverRating: Optional[float] = None
    Days_Policy_Accident: int
    Days_Policy_Claim: int
    PastNumberOfClaims: int
    AgeOfVehicle: int
    AgeOfPolicyHolder: int
    NumberOfSuppliments: int
    NumberOfCars: int
    Year: int
    claim_delay_approx: float

    # Make (one-hot, drop_first=True → Chevy is reference)
    Make_BMW: bool = False
    Make_Chevrolet: bool = False
    Make_Dodge: bool = False
    Make_Ferrari: bool = False
    Make_Ford: bool = False
    Make_Honda: bool = False
    Make_Jaguar: bool = False
    Make_Lexus: bool = False
    Make_Mazda: bool = False
    Make_Mecedes: bool = False
    Make_Mercury: bool = False
    Make_Nisson: bool = False
    Make_Pontiac: bool = False
    Make_Porche: bool = False
    Make_Saab: bool = False
    Make_Saturn: bool = False
    Make_Toyota: bool = False
    Make_VW: bool = False

    AccidentArea_Urban: bool = False
    Sex_Male: bool = False
    MaritalStatus_Married: bool = False
    MaritalStatus_Single: bool = False
    MaritalStatus_Widow: bool = False

    # columns whose names contain spaces — exposed via alias
    Fault_Third_Party: bool = Field(False, alias="Fault_Third Party")

    PolicyType_Sedan_Collision: bool = Field(False, alias="PolicyType_Sedan - Collision")
    PolicyType_Sedan_Liability: bool = Field(False, alias="PolicyType_Sedan - Liability")
    PolicyType_Sport_All_Perils: bool = Field(False, alias="PolicyType_Sport - All Perils")
    PolicyType_Sport_Collision: bool = Field(False, alias="PolicyType_Sport - Collision")
    PolicyType_Sport_Liability: bool = Field(False, alias="PolicyType_Sport - Liability")
    PolicyType_Utility_All_Perils: bool = Field(False, alias="PolicyType_Utility - All Perils")
    PolicyType_Utility_Collision: bool = Field(False, alias="PolicyType_Utility - Collision")
    PolicyType_Utility_Liability: bool = Field(False, alias="PolicyType_Utility - Liability")

    VehicleCategory_Sport: bool = False
    VehicleCategory_Utility: bool = False
    PoliceReportFiled_Yes: bool = False
    WitnessPresent_Yes: bool = False
    AgentType_Internal: bool = False

    AddressChange_Claim_2_to_3_years: bool = Field(False, alias="AddressChange_Claim_2 to 3 years")
    AddressChange_Claim_4_to_8_years: bool = Field(False, alias="AddressChange_Claim_4 to 8 years")
    AddressChange_Claim_no_change: bool = Field(False, alias="AddressChange_Claim_no change")
    AddressChange_Claim_under_6_months: bool = Field(False, alias="AddressChange_Claim_under 6 months")

    BasePolicy_Collision: bool = False
    BasePolicy_Liability: bool = False

    DayOfWeek_Monday: bool = False
    DayOfWeek_Saturday: bool = False
    DayOfWeek_Sunday: bool = False
    DayOfWeek_Thursday: bool = False
    DayOfWeek_Tuesday: bool = False
    DayOfWeek_Wednesday: bool = False

    DayOfWeekClaimed_Monday: bool = False
    DayOfWeekClaimed_Saturday: bool = False
    DayOfWeekClaimed_Sunday: bool = False
    DayOfWeekClaimed_Thursday: bool = False
    DayOfWeekClaimed_Tuesday: bool = False
    DayOfWeekClaimed_Wednesday: bool = False


@app.get("/")
def read_root():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: ClaimData):
    input_df = pd.DataFrame([data.model_dump(by_alias=True)])
    proba = model.predict_proba(input_df)[0][1]
    threshold = 0.091
    is_fraud = proba >= threshold
    return {"fraud_probability": float(proba), "is_fraud": bool(is_fraud)}