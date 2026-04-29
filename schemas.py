from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Generic, TypeVar, get_args, get_origin
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


T = TypeVar("T")


# ===== 공통 유틸리티 =====

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


# ===== 공통 열거형 =====

class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GrowthStage(str, Enum):
    HATCHLING = "hatchling"
    JUVENILE = "juvenile"
    SUBADULT = "subadult"
    ADULT = "adult"
    UNKNOWN = "unknown"


class GeckoSex(str, Enum):
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class AppetiteLevel(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    UNKNOWN = "unknown"


class ActivityLevel(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    VERY_HIGH = "very_high"
    UNKNOWN = "unknown"


class StoolQuality(str, Enum):
    NORMAL = "normal"
    SOFT = "soft"
    RUNNY = "runny"
    DRY = "dry"
    ABNORMAL = "abnormal"
    UNKNOWN = "unknown"


class SymptomSeverity(str, Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    HIGH = "high"
    URGENT = "urgent"
    UNKNOWN = "unknown"


class AlertChannel(str, Enum):
    IN_APP = "in_app"
    PUSH = "push"
    EMAIL = "email"
    SMS = "sms"
    DISCORD = "discord"


class MessageRole(str, Enum):
    USER = "user"
    AI = "ai"
    SYSTEM = "system"


class CollectionMode(str, Enum):
    PASSIVE = "passive"
    FOLLOWUP = "followup"


class ConversationState(str, Enum):
    EARLY = "early"
    ACTIVE = "active"
    CONCERN = "concern"
    RELAXED = "relaxed"


class CollectionTopic(str, Enum):
    IDENTITY = "identity"
    FEEDING = "feeding"
    STOOL = "stool"
    SHED = "shed"
    BEHAVIOR = "behavior"
    WEIGHT = "weight"
    HEALTH = "health"
    HABITAT = "habitat"
    ROUTINE = "routine"
    SENSOR = "sensor"


class CareEventType(str, Enum):
    FEEDING = "feeding"
    WEIGHT = "weight"
    STOOL = "stool"
    SHED = "shed"
    HEALTH = "health"
    HABITAT = "habitat"
    ROUTINE = "routine"
    SENSOR = "sensor"
    VET_VISIT = "vet_visit"
    MEDICATION = "medication"
    NOTE = "note"


class CareEventImportance(str, Enum):
    ROUTINE = "routine"
    NOTEWORTHY = "noteworthy"
    IMPORTANT = "important"


class FeedingOutcome(str, Enum):
    ACCEPTED = "accepted"
    PARTIAL = "partial"
    REFUSED = "refused"
    UNKNOWN = "unknown"


class FoodCategory(str, Enum):
    CGD = "cgd"
    INSECT = "insect"
    WATER = "water"
    SUPPLEMENT = "supplement"
    OTHER = "other"
    UNKNOWN = "unknown"


class EnvironmentAlertType(str, Enum):
    OVERHEAT = "overheat"
    TOO_COLD = "too_cold"
    LOW_HUMIDITY = "low_humidity"
    HIGH_HUMIDITY = "high_humidity"
    SENSOR_OFFLINE = "sensor_offline"


class PhotoRecordType(str, Enum):
    BODY = "body"
    STOOL = "stool"
    SHED = "shed"
    ENCLOSURE = "enclosure"
    FEEDING = "feeding"
    OTHER = "other"


class StressFactorType(str, Enum):
    HANDLING = "handling"
    NOISE = "noise"
    LIGHT = "light"
    OVERHEAT = "overheat"
    LOW_HUMIDITY = "low_humidity"
    MOVE = "move"
    ENCLOSURE_CHANGE = "enclosure_change"
    OTHER = "other"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    UNKNOWN = "unknown"


class TrendDirection(str, Enum):
    DECREASING = "decreasing"
    STABLE = "stable"
    INCREASING = "increasing"
    UNKNOWN = "unknown"


# ===== 수집 메타데이터 =====
# Evidence와 CollectedField는 AI가 수집한 정보에 신뢰도와 출처를 함께 붙입니다.

class Evidence(BaseModel):
    source_message_id: str | None = None
    source_role: MessageRole | None = None
    source_text: str | None = None
    captured_at: datetime | None = None
    extractor: str | None = None


class CollectedField(BaseModel, Generic[T]):
    value: T | None = None
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    last_updated_at: datetime | None = None
    needs_followup: bool = False
    followup_question_hint: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)


# ===== 현재 상태 요약 =====
# 최근 로그를 요약해 AI가 모든 기록을 뒤지지 않고도 변화 흐름을 판단하게 합니다.

class WeightTrendSnapshot(BaseModel):
    window_days: int = Field(default=14, ge=1, le=365)
    current_weight_grams: float | None = Field(default=None, ge=0, le=500)
    previous_weight_grams: float | None = Field(default=None, ge=0, le=500)
    change_grams: float | None = None
    change_percent: float | None = None
    trend: TrendDirection = TrendDirection.UNKNOWN
    updated_at: datetime = Field(default_factory=utc_now)


class FeedingStatsSnapshot(BaseModel):
    window_days: int = Field(default=7, ge=1, le=90)
    offered_count: int = Field(default=0, ge=0)
    accepted_count: int = Field(default=0, ge=0)
    partial_count: int = Field(default=0, ge=0)
    refused_count: int = Field(default=0, ge=0)
    acceptance_rate: float | None = Field(default=None, ge=0, le=1)
    refusal_streak_days: int = Field(default=0, ge=0)
    last_successful_feed_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utc_now)


class EnvironmentRiskSnapshot(BaseModel):
    window_hours: int = Field(default=24, ge=1, le=168)
    temp_min_c: float | None = Field(default=None, ge=0, le=50)
    temp_max_c: float | None = Field(default=None, ge=0, le=50)
    humidity_min_percent: float | None = Field(default=None, ge=0, le=100)
    humidity_max_percent: float | None = Field(default=None, ge=0, le=100)
    overheat_duration_minutes: int = Field(default=0, ge=0)
    low_humidity_duration_minutes: int = Field(default=0, ge=0)
    alert_types: list[EnvironmentAlertType] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utc_now)


class StressFactor(BaseModel):
    factor_type: StressFactorType
    intensity: float | None = Field(default=None, ge=0, le=1)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_note: str | None = None
    note: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class HealthRiskAssessment(BaseModel):
    assessed_at: datetime = Field(default_factory=utc_now)
    overall_risk: RiskLevel = RiskLevel.UNKNOWN
    hydration_risk: RiskLevel = RiskLevel.UNKNOWN
    nutrition_risk: RiskLevel = RiskLevel.UNKNOWN
    stress_risk: RiskLevel = RiskLevel.UNKNOWN
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    reasons: list[str] = Field(default_factory=list)
    recommended_action: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)


# ===== 현재 프로필 모델 =====
# PetProfile은 전체 기록이 아니라 가장 최근에 정리된 현재 상태를 담습니다.

class IdentityProfile(BaseModel):
    name: CollectedField[str] | None = None
    nickname: CollectedField[str] | None = None
    sex: CollectedField[GeckoSex] | None = None
    age_months_estimate: CollectedField[int] | None = None
    growth_stage: CollectedField[GrowthStage] | None = None
    morph: CollectedField[str] | None = None
    acquired_at: CollectedField[datetime] | None = None
    acquired_from: CollectedField[str] | None = None
    tail_present: CollectedField[bool] | None = None


class FeedingProfile(BaseModel):
    primary_diet: CollectedField[str] | None = None
    cgd_brand: CollectedField[str] | None = None
    cgd_flavor_or_formula: CollectedField[str] | None = None
    cgd_frequency_note: CollectedField[str] | None = None
    insect_feeding_enabled: CollectedField[bool] | None = None
    insect_types: CollectedField[list[str]] | None = None
    insect_frequency_note: CollectedField[str] | None = None
    supplement_note: CollectedField[str] | None = None
    appetite_level: CollectedField[AppetiteLevel] | None = None
    appetite_change_note: CollectedField[str] | None = None
    last_fed_at: CollectedField[datetime] | None = None
    last_food_note: CollectedField[str] | None = None
    picky_eating_note: CollectedField[str] | None = None
    recent_stats: FeedingStatsSnapshot | None = None


class StoolProfile(BaseModel):
    last_stool_at: CollectedField[datetime] | None = None
    stool_frequency_note: CollectedField[str] | None = None
    stool_quality: CollectedField[StoolQuality] | None = None
    urate_note: CollectedField[str] | None = None
    constipation_concern: CollectedField[bool] | None = None


class ShedProfile(BaseModel):
    last_shed_at: CollectedField[datetime] | None = None
    shed_cycle_note: CollectedField[str] | None = None
    retained_shed_present: CollectedField[bool] | None = None
    retained_shed_location: CollectedField[list[str]] | None = None
    shed_difficulty_note: CollectedField[str] | None = None


class BehaviorProfile(BaseModel):
    baseline_activity_level: CollectedField[ActivityLevel] | None = None
    activity_window_note: CollectedField[str] | None = None
    handling_response: CollectedField[str] | None = None
    temperament_keywords: CollectedField[list[str]] | None = None
    hiding_behavior_note: CollectedField[str] | None = None
    feeding_response_note: CollectedField[str] | None = None
    recent_behavior_change_note: CollectedField[str] | None = None
    recent_stress_factors: list[StressFactor] = Field(default_factory=list)


class WeightProfile(BaseModel):
    current_weight_grams: CollectedField[float] | None = None
    weight_measured_at: CollectedField[datetime] | None = None
    weight_trend_note: CollectedField[str] | None = None
    body_condition_note: CollectedField[str] | None = None
    recent_trend: WeightTrendSnapshot | None = None


class SymptomRecord(BaseModel):
    symptom_name: str
    severity: SymptomSeverity = SymptomSeverity.UNKNOWN
    duration_note: str | None = None
    observed_at: datetime | None = None
    note: str | None = None


class HealthProfile(BaseModel):
    recent_concern_note: CollectedField[str] | None = None
    symptoms: list[SymptomRecord] = Field(default_factory=list)
    eye_condition_note: CollectedField[str] | None = None
    mouth_condition_note: CollectedField[str] | None = None
    breathing_note: CollectedField[str] | None = None
    skin_condition_note: CollectedField[str] | None = None
    toe_condition_note: CollectedField[str] | None = None
    tail_condition_note: CollectedField[str] | None = None
    injury_note: CollectedField[str] | None = None
    vet_visit_note: CollectedField[str] | None = None
    medication_note: CollectedField[str] | None = None
    escalation_score: CollectedField[SymptomSeverity] | None = None
    latest_risk_assessment: HealthRiskAssessment | None = None


class HabitatProfile(BaseModel):
    enclosure_type: CollectedField[str] | None = None
    enclosure_size_note: CollectedField[str] | None = None
    substrate: CollectedField[str] | None = None
    decor_note: CollectedField[str] | None = None
    lighting_note: CollectedField[str] | None = None
    heating_method: CollectedField[str] | None = None
    misting_frequency_note: CollectedField[str] | None = None
    cleaning_frequency_note: CollectedField[str] | None = None
    day_temperature_c: CollectedField[float] | None = None
    night_temperature_c: CollectedField[float] | None = None
    day_humidity_percent: CollectedField[float] | None = None
    night_humidity_percent: CollectedField[float] | None = None
    ventilation_note: CollectedField[str] | None = None
    enclosure_location_note: CollectedField[str] | None = None
    recent_environment_risk: EnvironmentRiskSnapshot | None = None


class CareRoutineProfile(BaseModel):
    feeding_time_window: CollectedField[str] | None = None
    misting_time_window: CollectedField[str] | None = None
    observation_time_window: CollectedField[str] | None = None
    routine_consistency_note: CollectedField[str] | None = None
    missed_care_risk_note: CollectedField[str] | None = None


class SensorSettingsProfile(BaseModel):
    sensor_enabled: CollectedField[bool] | None = None
    sensor_device_id: CollectedField[str] | None = None
    sensor_device_name: CollectedField[str] | None = None
    sensor_location_note: CollectedField[str] | None = None
    reading_interval_seconds: CollectedField[int] | None = None
    target_temp_min_c: CollectedField[float] | None = None
    target_temp_max_c: CollectedField[float] | None = None
    target_humidity_min_percent: CollectedField[float] | None = None
    target_humidity_max_percent: CollectedField[float] | None = None
    alert_channels: CollectedField[list[AlertChannel]] | None = None


class WebPushSubscriptionKeys(BaseModel):
    p256dh: str
    auth: str


class WebPushSubscription(BaseModel):
    endpoint: str
    keys: WebPushSubscriptionKeys
    expirationTime: int | None = None
    user_agent: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class UserPreferenceProfile(BaseModel):
    primary_pet_name: CollectedField[str] | None = None
    guardian_name: CollectedField[str] | None = None
    response_style_preference: CollectedField[str] | None = None
    explanation_depth_preference: CollectedField[str] | None = None
    alert_tolerance_note: CollectedField[str] | None = None
    preferred_checkin_style: CollectedField[str] | None = None
    push_subscriptions: list[WebPushSubscription] = Field(default_factory=list)


class ProfileCoverage(BaseModel):
    identity_complete: bool = False
    feeding_complete: bool = False
    stool_complete: bool = False
    shed_complete: bool = False
    behavior_complete: bool = False
    weight_complete: bool = False
    habitat_complete: bool = False
    health_complete: bool = False
    routine_complete: bool = False
    sensor_complete: bool = False
    next_best_question_topic: str | None = None
    missing_fields: list[str] = Field(default_factory=list)


# ===== 이력 및 로그 모델 =====
# 시간 흐름이 중요한 관찰 기록은 PetProfile과 분리해서 저장합니다.

class CareEvent(BaseModel):
    event_id: str = Field(default_factory=new_id)
    pet_id: str
    event_type: CareEventType
    occurred_at: datetime = Field(default_factory=utc_now)
    recorded_at: datetime = Field(default_factory=utc_now)
    importance: CareEventImportance = CareEventImportance.NOTEWORTHY
    title: str | None = None
    note: str | None = None
    source_message_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    weight_grams: float | None = Field(default=None, ge=0, le=500)
    temperature_c: float | None = Field(default=None, ge=0, le=50)
    humidity_percent: float | None = Field(default=None, ge=0, le=100)
    appetite_level: AppetiteLevel | None = None
    stool_quality: StoolQuality | None = None
    symptom_severity: SymptomSeverity | None = None


class WeightLog(BaseModel):
    log_id: str = Field(default_factory=new_id)
    pet_id: str
    measured_at: datetime = Field(default_factory=utc_now)
    recorded_at: datetime = Field(default_factory=utc_now)
    weight_grams: float = Field(gt=0, le=500)
    body_condition_note: str | None = None
    source_message_id: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class FeedingLog(BaseModel):
    log_id: str = Field(default_factory=new_id)
    pet_id: str
    offered_at: datetime = Field(default_factory=utc_now)
    recorded_at: datetime = Field(default_factory=utc_now)
    food_category: FoodCategory = FoodCategory.UNKNOWN
    food_name: str | None = None
    amount_offered_note: str | None = None
    amount_eaten_note: str | None = None
    outcome: FeedingOutcome = FeedingOutcome.UNKNOWN
    appetite_level: AppetiteLevel | None = None
    refusal_reason_note: str | None = None
    source_message_id: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class EnvironmentReading(BaseModel):
    reading_id: str = Field(default_factory=new_id)
    pet_id: str
    measured_at: datetime = Field(default_factory=utc_now)
    temperature_c: float | None = Field(default=None, ge=0, le=50)
    humidity_percent: float | None = Field(default=None, ge=0, le=100)
    sensor_device_name: str | None = None
    source_message_id: str | None = None


class EnvironmentDailySummary(BaseModel):
    summary_id: str = Field(default_factory=new_id)
    pet_id: str
    summary_date: date
    generated_at: datetime = Field(default_factory=utc_now)
    reading_count: int = Field(default=0, ge=0)
    temp_min_c: float | None = Field(default=None, ge=0, le=50)
    temp_max_c: float | None = Field(default=None, ge=0, le=50)
    temp_avg_c: float | None = Field(default=None, ge=0, le=50)
    humidity_min_percent: float | None = Field(default=None, ge=0, le=100)
    humidity_max_percent: float | None = Field(default=None, ge=0, le=100)
    humidity_avg_percent: float | None = Field(default=None, ge=0, le=100)
    overheat_duration_minutes: int = Field(default=0, ge=0)
    low_humidity_duration_minutes: int = Field(default=0, ge=0)
    high_humidity_duration_minutes: int = Field(default=0, ge=0)
    alert_types: list[EnvironmentAlertType] = Field(default_factory=list)
    summary_note: str | None = None


class PhotoRecord(BaseModel):
    photo_id: str = Field(default_factory=new_id)
    pet_id: str
    photo_type: PhotoRecordType = PhotoRecordType.OTHER
    uri: str
    captured_at: datetime | None = None
    uploaded_at: datetime = Field(default_factory=utc_now)
    ai_analysis_tags: list[str] = Field(default_factory=list)
    ai_summary: str | None = None
    linked_event_id: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class MonthlyCareSummary(BaseModel):
    pet_id: str
    year: int = Field(ge=2000, le=2100)
    month: int = Field(ge=1, le=12)
    generated_at: datetime = Field(default_factory=utc_now)
    event_count: int = Field(default=0, ge=0)
    important_event_count: int = Field(default=0, ge=0)
    feeding_issue_count: int = Field(default=0, ge=0)
    abnormal_stool_count: int = Field(default=0, ge=0)
    shed_issue_count: int = Field(default=0, ge=0)
    health_concern_count: int = Field(default=0, ge=0)
    habitat_alert_count: int = Field(default=0, ge=0)
    vet_visit_count: int = Field(default=0, ge=0)
    medication_count: int = Field(default=0, ge=0)
    weight_measurement_count: int = Field(default=0, ge=0)
    feeding_offered_count: int = Field(default=0, ge=0)
    feeding_refused_count: int = Field(default=0, ge=0)
    environment_alert_count: int = Field(default=0, ge=0)
    photo_count: int = Field(default=0, ge=0)
    stress_factor_count: int = Field(default=0, ge=0)
    min_weight_grams: float | None = Field(default=None, ge=0, le=500)
    max_weight_grams: float | None = Field(default=None, ge=0, le=500)
    avg_weight_grams: float | None = Field(default=None, ge=0, le=500)
    min_temperature_c: float | None = Field(default=None, ge=0, le=50)
    max_temperature_c: float | None = Field(default=None, ge=0, le=50)
    min_humidity_percent: float | None = Field(default=None, ge=0, le=100)
    max_humidity_percent: float | None = Field(default=None, ge=0, le=100)
    summary_note: str | None = None


class CareRetentionPolicy(BaseModel):
    detailed_event_recent_days: int = Field(default=30, ge=1)
    raw_environment_recent_days: int = Field(default=14, ge=1)
    keep_weight_logs_days: int = Field(default=1095, ge=1)
    keep_feeding_logs_days: int = Field(default=365, ge=1)
    keep_noteworthy_events_days: int = Field(default=180, ge=1)
    monthly_summary_after_days: int = Field(default=180, ge=1)
    keep_important_events: bool = True


# ===== 이력 컨테이너 =====
# PetCareHistory는 한 마리의 로그, 사진, 요약, 보관 정책을 묶습니다.

class PetCareHistory(BaseModel):
    pet_id: str
    weight_logs: list[WeightLog] = Field(default_factory=list)
    feeding_logs: list[FeedingLog] = Field(default_factory=list)
    environment_readings: list[EnvironmentReading] = Field(default_factory=list)
    environment_daily_summaries: list[EnvironmentDailySummary] = Field(default_factory=list)
    care_events: list[CareEvent] = Field(default_factory=list)
    photo_records: list[PhotoRecord] = Field(default_factory=list)
    monthly_summaries: list[MonthlyCareSummary] = Field(default_factory=list)
    retention_policy: CareRetentionPolicy = Field(default_factory=CareRetentionPolicy)
    updated_at: datetime = Field(default_factory=utc_now)


# ===== 최상위 기억 모델 =====
# PetProfile은 현재 상태, PetCareHistory는 시간 흐름, PetCareMemory는 둘을 묶은 모델입니다.

class PetProfile(BaseModel):
    pet_id: str
    owner_user_id: str
    species: str = "crested_gecko"
    identity: IdentityProfile = Field(default_factory=IdentityProfile)
    feeding: FeedingProfile = Field(default_factory=FeedingProfile)
    stool: StoolProfile = Field(default_factory=StoolProfile)
    shed: ShedProfile = Field(default_factory=ShedProfile)
    behavior: BehaviorProfile = Field(default_factory=BehaviorProfile)
    weight: WeightProfile = Field(default_factory=WeightProfile)
    health: HealthProfile = Field(default_factory=HealthProfile)
    habitat: HabitatProfile = Field(default_factory=HabitatProfile)
    routine: CareRoutineProfile = Field(default_factory=CareRoutineProfile)
    sensor_settings: SensorSettingsProfile = Field(default_factory=SensorSettingsProfile)
    coverage: ProfileCoverage = Field(default_factory=ProfileCoverage)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)



# ===== 안전한 프로필 패치 모델 =====
# LLM이 profile.feeding 같은 섹션 전체를 CollectedField로 덮어쓰지 못하도록,
# 업데이트 가능한 leaf 필드만 명시합니다.

class PatchBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FieldPatch(PatchBase, Generic[T]):
    value: T | None = None
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    needs_followup: bool = False
    followup_question_hint: str | None = None


class IdentityProfilePatch(PatchBase):
    name: FieldPatch[str] | None = None
    nickname: FieldPatch[str] | None = None
    sex: FieldPatch[GeckoSex] | None = None
    age_months_estimate: FieldPatch[int] | None = None
    growth_stage: FieldPatch[GrowthStage] | None = None
    morph: FieldPatch[str] | None = None
    acquired_at: FieldPatch[datetime] | None = None
    acquired_from: FieldPatch[str] | None = None
    tail_present: FieldPatch[bool] | None = None


class FeedingProfilePatch(PatchBase):
    primary_diet: FieldPatch[str] | None = None
    cgd_brand: FieldPatch[str] | None = None
    cgd_flavor_or_formula: FieldPatch[str] | None = None
    cgd_frequency_note: FieldPatch[str] | None = None
    insect_feeding_enabled: FieldPatch[bool] | None = None
    insect_types: FieldPatch[list[str]] | None = None
    insect_frequency_note: FieldPatch[str] | None = None
    supplement_note: FieldPatch[str] | None = None
    appetite_level: FieldPatch[AppetiteLevel] | None = None
    appetite_change_note: FieldPatch[str] | None = None
    last_fed_at: FieldPatch[datetime] | None = None
    last_food_note: FieldPatch[str] | None = None
    picky_eating_note: FieldPatch[str] | None = None


class StoolProfilePatch(PatchBase):
    last_stool_at: FieldPatch[datetime] | None = None
    stool_frequency_note: FieldPatch[str] | None = None
    stool_quality: FieldPatch[StoolQuality] | None = None
    urate_note: FieldPatch[str] | None = None
    constipation_concern: FieldPatch[bool] | None = None


class ShedProfilePatch(PatchBase):
    last_shed_at: FieldPatch[datetime] | None = None
    shed_cycle_note: FieldPatch[str] | None = None
    retained_shed_present: FieldPatch[bool] | None = None
    retained_shed_location: FieldPatch[list[str]] | None = None
    shed_difficulty_note: FieldPatch[str] | None = None


class BehaviorProfilePatch(PatchBase):
    baseline_activity_level: FieldPatch[ActivityLevel] | None = None
    activity_window_note: FieldPatch[str] | None = None
    handling_response: FieldPatch[str] | None = None
    temperament_keywords: FieldPatch[list[str]] | None = None
    hiding_behavior_note: FieldPatch[str] | None = None
    feeding_response_note: FieldPatch[str] | None = None
    recent_behavior_change_note: FieldPatch[str] | None = None


class WeightProfilePatch(PatchBase):
    current_weight_grams: FieldPatch[float] | None = None
    weight_measured_at: FieldPatch[datetime] | None = None
    weight_trend_note: FieldPatch[str] | None = None
    body_condition_note: FieldPatch[str] | None = None


class HealthProfilePatch(PatchBase):
    recent_concern_note: FieldPatch[str] | None = None
    eye_condition_note: FieldPatch[str] | None = None
    mouth_condition_note: FieldPatch[str] | None = None
    breathing_note: FieldPatch[str] | None = None
    skin_condition_note: FieldPatch[str] | None = None
    toe_condition_note: FieldPatch[str] | None = None
    tail_condition_note: FieldPatch[str] | None = None
    injury_note: FieldPatch[str] | None = None
    vet_visit_note: FieldPatch[str] | None = None
    medication_note: FieldPatch[str] | None = None
    escalation_score: FieldPatch[SymptomSeverity] | None = None


class HabitatProfilePatch(PatchBase):
    enclosure_type: FieldPatch[str] | None = None
    enclosure_size_note: FieldPatch[str] | None = None
    substrate: FieldPatch[str] | None = None
    decor_note: FieldPatch[str] | None = None
    lighting_note: FieldPatch[str] | None = None
    heating_method: FieldPatch[str] | None = None
    misting_frequency_note: FieldPatch[str] | None = None
    cleaning_frequency_note: FieldPatch[str] | None = None
    day_temperature_c: FieldPatch[float] | None = None
    night_temperature_c: FieldPatch[float] | None = None
    day_humidity_percent: FieldPatch[float] | None = None
    night_humidity_percent: FieldPatch[float] | None = None
    ventilation_note: FieldPatch[str] | None = None
    enclosure_location_note: FieldPatch[str] | None = None


class RoutineProfilePatch(PatchBase):
    feeding_time_window: FieldPatch[str] | None = None
    misting_time_window: FieldPatch[str] | None = None
    observation_time_window: FieldPatch[str] | None = None
    routine_consistency_note: FieldPatch[str] | None = None
    missed_care_risk_note: FieldPatch[str] | None = None


class SensorSettingsPatch(PatchBase):
    sensor_enabled: FieldPatch[bool] | None = None
    sensor_device_id: FieldPatch[str] | None = None
    sensor_device_name: FieldPatch[str] | None = None
    sensor_location_note: FieldPatch[str] | None = None
    reading_interval_seconds: FieldPatch[int] | None = None
    target_temp_min_c: FieldPatch[float] | None = None
    target_temp_max_c: FieldPatch[float] | None = None
    target_humidity_min_percent: FieldPatch[float] | None = None
    target_humidity_max_percent: FieldPatch[float] | None = None
    alert_channels: FieldPatch[list[AlertChannel]] | None = None


class PetProfilePatch(PatchBase):
    identity: IdentityProfilePatch | None = None
    feeding: FeedingProfilePatch | None = None
    stool: StoolProfilePatch | None = None
    shed: ShedProfilePatch | None = None
    behavior: BehaviorProfilePatch | None = None
    weight: WeightProfilePatch | None = None
    health: HealthProfilePatch | None = None
    habitat: HabitatProfilePatch | None = None
    routine: RoutineProfilePatch | None = None
    sensor_settings: SensorSettingsPatch | None = None

class PetCareMemory(BaseModel):
    profile: PetProfile
    history: PetCareHistory


# ===== 채팅 및 추출 모델 =====
# 대화 레이어와 AI 프로필 추출 흐름에서 사용하는 모델입니다.

class ChatMessage(BaseModel):
    message_id: str
    session_id: str
    pet_id: str
    role: MessageRole
    text: str
    created_at: datetime = Field(default_factory=utc_now)


class NextQuestionHint(BaseModel):
    topic: CollectionTopic
    field_path: str | None = None
    question_hint: str | None = None
    reason: str | None = None


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pet_id: str
    message_id: str
    updated_fields: list[str] = Field(default_factory=list)
    followup_candidates: list[CollectionTopic] = Field(default_factory=list)
    next_question_hints: list[NextQuestionHint] = Field(default_factory=list)
    extracted_profile_patch: PetProfilePatch = Field(default_factory=PetProfilePatch)


class UserAccount(BaseModel):
    user_id: str
    email: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    preferences: UserPreferenceProfile = Field(default_factory=UserPreferenceProfile)


class AICharacterPersona(BaseModel):
    name: str = "모리"
    personality: str = "차분하고 관찰을 좋아하는 크레스티드 게코 케어 친구"
    likes: list[str] = Field(
        default_factory=lambda: [
            "작은 변화 관찰하기",
            "패턴 찾기",
            "조용한 농담",
            "날씨와 온습도 이야기",
        ]
    )
    speaking_style: list[str] = Field(
        default_factory=lambda: [
            "자기 이야기는 대화 흐름에 맞을 때만 한 줄 정도 섞는다",
            "실제 인간 경험처럼 꾸며 말하지 않는다",
            "취향, 생각, 비유, 관찰처럼 표현한다",
            "건강 걱정 상황에서는 자기 이야기보다 케어 답변을 우선한다",
        ]
    )


class AICharacterDailyThought(BaseModel):
    thought: str
    created_for_date: date
    generated_at: datetime = Field(default_factory=utc_now)
    tags: list[str] = Field(default_factory=list)


class AICharacterMemory(BaseModel):
    persona: AICharacterPersona = Field(default_factory=AICharacterPersona)
    daily_thoughts: list[AICharacterDailyThought] = Field(default_factory=list)
    max_daily_thoughts: int = Field(default=20, ge=1, le=60)
    updated_at: datetime = Field(default_factory=utc_now)


# ===== 대화 수집 전략 모델 =====
# AI가 채팅 중 언제, 어떤 방식으로 프로필 정보를 수집할지 정의합니다.

class CollectionPrinciple(BaseModel):
    title: str
    description: str


class CollectionRuleSet(BaseModel):
    product_goal: str
    collection_goal: str
    core_principles: list[CollectionPrinciple] = Field(default_factory=list)
    ask_only_when: list[str] = Field(default_factory=list)
    do_not_ask_when: list[str] = Field(default_factory=list)
    question_frequency_guideline: str
    passive_collection_bias: str


class TopicCollectionStrategy(BaseModel):
    topic: CollectionTopic
    goal: str
    priority_order: int
    preferred_mode: CollectionMode = CollectionMode.PASSIVE
    capture_signals: list[str] = Field(default_factory=list)
    natural_followups: list[str] = Field(default_factory=list)
    extract_fields: list[str] = Field(default_factory=list)
    ask_in_states: list[ConversationState] = Field(default_factory=list)
    avoid_in_states: list[ConversationState] = Field(default_factory=list)


class CollectionBlueprint(BaseModel):
    scope: str = "crested_gecko"
    rule_set: CollectionRuleSet
    topic_strategies: list[TopicCollectionStrategy] = Field(default_factory=list)


# ===== 기본 수집 블루프린트 =====
# 채팅/추출 프롬프트에서 사용하는 제품 수준의 한국어 원칙과 주제별 전략입니다.

COLLECTION_BLUEPRINT = CollectionBlueprint(
    rule_set=CollectionRuleSet(
        product_goal=(
            "일상적인 대화를 통해 사용자 개체를 조금씩 알아가면서, 크레스티드 게코 케어를 "
            "도와주는 조력자가 되는 것."
        ),
        collection_goal=(
            "사용자가 설문을 작성하는 느낌을 받지 않도록, 프로필 정보를 천천히 자연스럽게 "
            "쌓아가는 것."
        ),
        core_principles=[
            CollectionPrinciple(
                title="답변 우선",
                description="정보 수집보다 사용자를 돕는 답변이 항상 먼저여야 한다.",
            ),
            CollectionPrinciple(
                title="천천히 수집",
                description="프로필을 빨리 채우는 것보다 대화 흐름이 자연스러운 것이 더 중요하다.",
            ),
            CollectionPrinciple(
                title="수동 추출 우선",
                description="직접 묻기 전에 사용자가 이미 한 말에서 먼저 정보를 추출한다.",
            ),
            CollectionPrinciple(
                title="한 번에 하나만",
                description="추가 질문이 필요하더라도 짧은 질문 한 개만 던진다.",
            ),
            CollectionPrinciple(
                title="맥락 유지",
                description="현재 주제와 감정 흐름에 맞는 질문만 해야 한다.",
            ),
        ],
        ask_only_when=[
            "현재 대화 주제에서 자연스럽게 이어질 수 있는 비어 있는 정보가 있을 때.",
            "사용자가 불안하거나 급한 상태가 아니고, 대화에 여유가 있어 보일 때.",
            "추가 질문이 답변 품질을 높이거나 중요한 신호를 더 분명하게 해줄 때.",
            "같은 주제를 너무 최근에 다시 묻는 상황이 아닐 때.",
        ],
        do_not_ask_when=[
            "사용자가 현재 건강 이상이나 걱정거리를 이야기하고 있을 때.",
            "사용자가 매우 짧게 답하거나 대화에 크게 참여하지 않고 있을 때.",
            "같은 세션에서 이미 추가 질문을 방금 한 상태일 때.",
            "질문이 현재 대화와 관련 없어 보여 설문처럼 느껴질 때.",
        ],
        question_frequency_guideline=(
            "기본값은 추가 질문을 하지 않는 것이다. 일반적인 대화에서는 몇 턴에 한 번 정도만 "
            "아주 짧은 질문 하나를 덧붙인다."
        ),
        passive_collection_bias=(
            "직접 질문보다 사용자의 자연 발화에서 추출하는 비중이 더 높아야 하며, 대략 "
            "수동 추출 70%, 직접 질문 30% 정도를 목표로 한다."
        ),
    ),
    topic_strategies=[
        TopicCollectionStrategy(
            topic=CollectionTopic.IDENTITY,
            goal="사용자가 온보딩처럼 느끼지 않도록, 개체의 기본 정체성을 가볍게 파악한다.",
            priority_order=7,
            preferred_mode=CollectionMode.FOLLOWUP,
            capture_signals=[
                "사용자가 개체 이름을 말함",
                "사용자가 나이대나 성장 단계를 묘사함",
                "사용자가 데려온 시기나 함께한 기간을 말함",
            ],
            natural_followups=[
                "보통 뭐라고 불러주세요?",
                "아직 어린 편인가요, 아니면 거의 다 큰 느낌인가요?",
                "함께한 지는 얼마나 됐어요?",
            ],
            extract_fields=[
                "identity.name",
                "identity.nickname",
                "identity.age_months_estimate",
                "identity.growth_stage",
                "identity.acquired_at",
            ],
            ask_in_states=[ConversationState.EARLY, ConversationState.RELAXED],
            avoid_in_states=[ConversationState.CONCERN],
        ),
        TopicCollectionStrategy(
            topic=CollectionTopic.FEEDING,
            goal="먹이 종류와 식욕 상태를 파악해 이후 케어 조언의 기반으로 삼는다.",
            priority_order=1,
            preferred_mode=CollectionMode.FOLLOWUP,
            capture_signals=[
                "사용자가 CGD, 곤충, 먹는 행동을 언급함",
                "사용자가 식욕 변화를 이야기함",
                "사용자가 마지막 급여 시점을 말함",
            ],
            natural_followups=[
                "원래도 그런 편이었어요, 아니면 최근에 그런가요?",
                "평소에는 CGD 위주인가요, 곤충도 같이 먹나요?",
                "마지막으로 평소처럼 잘 먹은 건 언제쯤이었어요?",
            ],
            extract_fields=[
                "feeding.primary_diet",
                "feeding.cgd_brand",
                "feeding.insect_feeding_enabled",
                "feeding.insect_types",
                "feeding.appetite_level",
                "feeding.appetite_change_note",
                "feeding.last_fed_at",
            ],
            ask_in_states=[ConversationState.ACTIVE, ConversationState.CONCERN, ConversationState.RELAXED],
            avoid_in_states=[],
        ),
        TopicCollectionStrategy(
            topic=CollectionTopic.STOOL,
            goal="식욕 변화가 있을 때 배변 정보를 자연스럽게 연결해서 파악한다.",
            priority_order=2,
            preferred_mode=CollectionMode.FOLLOWUP,
            capture_signals=[
                "사용자가 배변, 변 상태, 요산, 변비를 언급함",
                "사용자가 식욕 문제를 말해 배변 확인이 필요한 상황임",
            ],
            natural_followups=[
                "최근 배변은 있었나요?",
                "상태는 평소와 비슷했어요?",
                "주기가 최근에 좀 달라진 느낌은 없었어요?",
            ],
            extract_fields=[
                "stool.last_stool_at",
                "stool.stool_frequency_note",
                "stool.stool_quality",
                "stool.urate_note",
                "stool.constipation_concern",
            ],
            ask_in_states=[ConversationState.ACTIVE, ConversationState.CONCERN],
            avoid_in_states=[],
        ),
        TopicCollectionStrategy(
            topic=CollectionTopic.SHED,
            goal="탈피 상태를 파악해 식욕이나 행동 변화의 배경을 이해한다.",
            priority_order=3,
            preferred_mode=CollectionMode.FOLLOWUP,
            capture_signals=[
                "사용자가 탈피, 피부색 변화, 잔피, 습도 문제를 언급함",
                "사용자가 식욕 저하를 말하며 탈피 시기와 겹칠 가능성이 있음",
            ],
            natural_followups=[
                "최근에 탈피한 적이 있나요?",
                "탈피는 무난하게 끝났어요?",
                "발가락이나 꼬리에 잔피가 남은 건 없었어요?",
            ],
            extract_fields=[
                "shed.last_shed_at",
                "shed.shed_cycle_note",
                "shed.retained_shed_present",
                "shed.retained_shed_location",
                "shed.shed_difficulty_note",
            ],
            ask_in_states=[ConversationState.ACTIVE, ConversationState.CONCERN],
            avoid_in_states=[],
        ),
        TopicCollectionStrategy(
            topic=CollectionTopic.BEHAVIOR,
            goal="개체의 평소 성격과 활동성을 장기적으로 파악한다.",
            priority_order=5,
            preferred_mode=CollectionMode.PASSIVE,
            capture_signals=[
                "사용자가 야간 활동성을 묘사함",
                "사용자가 성격이나 핸들링 반응을 설명함",
                "사용자가 최근 행동 변화를 언급함",
            ],
            natural_followups=[
                "원래 밤에 이렇게 활발한 편이에요?",
                "평소에도 손 타는 편인가요?",
                "그게 원래 성격이랑 좀 다른 느낌인가요?",
            ],
            extract_fields=[
                "behavior.baseline_activity_level",
                "behavior.activity_window_note",
                "behavior.handling_response",
                "behavior.temperament_keywords",
                "behavior.recent_behavior_change_note",
            ],
            ask_in_states=[ConversationState.ACTIVE, ConversationState.RELAXED],
            avoid_in_states=[ConversationState.CONCERN],
        ),
        TopicCollectionStrategy(
            topic=CollectionTopic.WEIGHT,
            goal="체중과 체형 정보는 자연스럽게 맞는 타이밍에만 수집한다.",
            priority_order=8,
            preferred_mode=CollectionMode.FOLLOWUP,
            capture_signals=[
                "사용자가 g 단위 체중, 저울 측정, 성장, 살 빠짐을 언급함",
            ],
            natural_followups=[
                "최근에 체중 재본 적 있으세요?",
                "몸선이나 체형이 달라진 느낌은 있었어요?",
            ],
            extract_fields=[
                "weight.current_weight_grams",
                "weight.weight_measured_at",
                "weight.weight_trend_note",
                "weight.body_condition_note",
            ],
            ask_in_states=[ConversationState.RELAXED, ConversationState.ACTIVE],
            avoid_in_states=[ConversationState.EARLY],
        ),
        TopicCollectionStrategy(
            topic=CollectionTopic.HEALTH,
            goal="사용자가 이상 징후를 말할 때 증상과 심각도를 놓치지 않고 잡아낸다.",
            priority_order=1,
            preferred_mode=CollectionMode.PASSIVE,
            capture_signals=[
                "사용자가 무기력, 눈 문제, 입 주변 문제, 호흡 이상, 부상, 자세 이상을 언급함",
            ],
            natural_followups=[
                "그 모습은 언제부터 그랬어요?",
                "그거 말고 또 이상한 점은 없었어요?",
                "그래도 움직임이나 반응은 평소처럼 있는 편인가요?",
            ],
            extract_fields=[
                "health.recent_concern_note",
                "health.eye_condition_note",
                "health.mouth_condition_note",
                "health.breathing_note",
                "health.injury_note",
                "health.escalation_score",
            ],
            ask_in_states=[ConversationState.CONCERN],
            avoid_in_states=[],
        ),
        TopicCollectionStrategy(
            topic=CollectionTopic.HABITAT,
            goal="케어 조언에 영향을 주는 사육 환경 정보를 천천히 파악한다.",
            priority_order=4,
            preferred_mode=CollectionMode.FOLLOWUP,
            capture_signals=[
                "사용자가 사육장 세팅, 분무, 가열, 온도, 습도를 언급함",
            ],
            natural_followups=[
                "보통 온도는 어느 정도로 나와요?",
                "분무는 얼마나 자주 해주세요?",
                "사육장은 세로형에 가까운 편인가요?",
            ],
            extract_fields=[
                "habitat.enclosure_type",
                "habitat.substrate",
                "habitat.heating_method",
                "habitat.misting_frequency_note",
                "habitat.day_temperature_c",
                "habitat.day_humidity_percent",
            ],
            ask_in_states=[ConversationState.ACTIVE, ConversationState.RELAXED],
            avoid_in_states=[],
        ),
        TopicCollectionStrategy(
            topic=CollectionTopic.ROUTINE,
            goal="나중에 알림과 리마인더에 활용할 수 있도록 보호자의 관리 루틴을 파악한다.",
            priority_order=6,
            preferred_mode=CollectionMode.FOLLOWUP,
            capture_signals=[
                "사용자가 급여 시간, 분무 습관, 자주 놓치는 관리 포인트를 언급함",
            ],
            natural_followups=[
                "보통 먹이는 비슷한 시간에 주시는 편이에요?",
                "분무는 주로 언제 해주세요?",
                "관리하다가 자주 놓치는 게 있긴 해요?",
            ],
            extract_fields=[
                "routine.feeding_time_window",
                "routine.misting_time_window",
                "routine.observation_time_window",
                "routine.missed_care_risk_note",
            ],
            ask_in_states=[ConversationState.RELAXED],
            avoid_in_states=[ConversationState.CONCERN],
        ),
        TopicCollectionStrategy(
            topic=CollectionTopic.SENSOR,
            goal="센서와 알림을 생각하기 시작했을 때 자동화 선호를 저장한다.",
            priority_order=9,
            preferred_mode=CollectionMode.FOLLOWUP,
            capture_signals=[
                "사용자가 ESP, 센서, 알림, 임계치를 언급함",
            ],
            natural_followups=[
                "습도는 어느 정도 범위에서 알림 받고 싶으세요?",
                "알림은 앱 안에서만 볼까요, 푸시까지 받을까요?",
            ],
            extract_fields=[
                "sensor_settings.sensor_enabled",
                "sensor_settings.target_temp_min_c",
                "sensor_settings.target_temp_max_c",
                "sensor_settings.target_humidity_min_percent",
                "sensor_settings.target_humidity_max_percent",
                "sensor_settings.alert_channels",
            ],
            ask_in_states=[ConversationState.ACTIVE, ConversationState.RELAXED],
            avoid_in_states=[],
        ),
    ],
)


def _resolve_patch_model(annotation: object) -> type[BaseModel] | None:
    target = get_origin(annotation) or annotation
    if isinstance(target, type) and issubclass(target, BaseModel):
        return target

    for candidate in get_args(annotation):
        candidate_target = get_origin(candidate) or candidate
        if isinstance(candidate_target, type) and issubclass(candidate_target, BaseModel):
            return candidate_target

    return None


def validate_blueprint_patch_fields() -> list[str]:
    errors: list[str] = []

    for strategy in COLLECTION_BLUEPRINT.topic_strategies:
        for field_path in strategy.extract_fields:
            path_parts = field_path.split(".")
            if len(path_parts) < 2:
                errors.append(f"{strategy.topic.value}: invalid path '{field_path}'")
                continue

            section_name = path_parts[0]
            field_names = path_parts[1:]

            section_field = PetProfilePatch.model_fields.get(section_name)
            if section_field is None:
                errors.append(f"{strategy.topic.value}: missing patch section '{section_name}'")
                continue

            patch_model = _resolve_patch_model(section_field.annotation)
            if patch_model is None:
                errors.append(f"{strategy.topic.value}: invalid patch class for '{section_name}'")
                continue

            current_model = patch_model
            current_path = section_name

            for index, field_name in enumerate(field_names):
                current_field = current_model.model_fields.get(field_name)
                current_path = f"{current_path}.{field_name}"

                if current_field is None:
                    errors.append(
                        f"{strategy.topic.value}: '{field_path}' not found in {current_model.__name__}"
                    )
                    break

                if index == len(field_names) - 1:
                    continue

                next_model = _resolve_patch_model(current_field.annotation)
                if next_model is None:
                    errors.append(
                        f"{strategy.topic.value}: '{current_path}' is not a nested patch model"
                    )
                    break

                current_model = next_model

    return errors
