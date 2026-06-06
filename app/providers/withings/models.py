from pydantic import BaseModel


class WithingsMeasureType:
    WEIGHT_KG = 1
    HEIGHT_M = 4
    FAT_FREE_MASS_KG = 5
    FAT_RATIO_PCT = 6
    FAT_MASS_KG = 8
    DIASTOLIC_BP = 9
    SYSTOLIC_BP = 10
    HEART_PULSE = 11
    MUSCLE_MASS_KG = 76
    HYDRATION_KG = 77
    BONE_MASS_KG = 88


class WithingsMeasure(BaseModel):
    value: int
    type: int
    unit: int

    @property
    def real_value(self) -> float:
        """Convert Withings internal representation to actual value.

        Withings stores measures as ``value * 10^unit``.
        """
        return self.value * (10**self.unit)


class WithingsMeasureGroup(BaseModel):
    grpid: int
    attrib: int
    date: int  # epoch timestamp
    category: int
    measures: list[WithingsMeasure]

    def get_measure(self, measure_type: int) -> float | None:
        for m in self.measures:
            if m.type == measure_type:
                return m.real_value
        return None


class WithingsMeasureBody(BaseModel):
    updatetime: int | None = None
    timezone: str | None = None
    measuregrps: list[WithingsMeasureGroup] = []


class WithingsMeasureResponse(BaseModel):
    status: int
    body: WithingsMeasureBody


class WithingsSleepSeries(BaseModel):
    id: int | None = None
    startdate: int  # epoch timestamp
    enddate: int  # epoch timestamp
    date: str | None = None
    model: int | None = None
    model_id: int | None = None
    deepsleepduration: int | None = None  # seconds
    lightsleepduration: int | None = None  # seconds
    remsleepduration: int | None = None  # seconds
    wakeupduration: int | None = None  # seconds
    hr_average: int | None = None
    hr_min: int | None = None
    hr_max: int | None = None


class WithingsSleepBody(BaseModel):
    series: list[WithingsSleepSeries] = []
    more: bool = False


class WithingsSleepResponse(BaseModel):
    status: int
    body: WithingsSleepBody


class WithingsTokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str = "Bearer"
    userid: int | None = None
    scope: str | None = None
