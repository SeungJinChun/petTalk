from __future__ import annotations

import json
import logging
import os
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest, urlopen
from urllib.parse import urlencode
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ValidationError

from auth_context import resolve_auth_user
from db_store import (
    append_chat_message as db_append_chat_message,
    append_extraction as db_append_extraction,
    delete_user_account_data as db_delete_user_account_data,
    load_character_memory as db_load_character_memory,
    load_chat_history as db_load_chat_history,
    load_pet_history as db_load_pet_history,
    load_sensor_link_target_by_device_id as db_load_sensor_link_target_by_device_id,
    load_pet_profile as db_load_pet_profile,
    load_recent_extractions as db_load_recent_extractions,
    load_user as db_load_user,
    save_character_memory as db_save_character_memory,
    save_chat_history as db_save_chat_history,
    save_pet_history as db_save_pet_history,
    save_pet_profile as db_save_pet_profile,
    save_user as db_save_user,
)

from schemas import (
    AICharacterDailyThought,
    AICharacterMemory,
    AppetiteLevel,
    COLLECTION_BLUEPRINT,
    CareEvent,
    CareEventImportance,
    CareEventType,
    ChatMessage,
    CollectedField,
    CollectionTopic,
    ConfidenceLevel,
    ConversationState,
    Evidence,
    EnvironmentAlertType,
    EnvironmentDailySummary,
    EnvironmentReading,
    EnvironmentRiskSnapshot,
    ExtractionResult,
    FeedingLog,
    FeedingOutcome,
    FeedingStatsSnapshot,
    FoodCategory,
    MessageRole,
    PetCareHistory,
    PetProfile,
    SymptomSeverity,
    TrendDirection,
    UserAccount,
    WebPushSubscription,
    WebPushSubscriptionKeys,
    WeightLog,
    WeightTrendSnapshot,
    utc_now,
    validate_blueprint_patch_fields,
)


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
logger = logging.getLogger(__name__)

BLUEPRINT_PATCH_FIELD_ERRORS = validate_blueprint_patch_fields()
if BLUEPRINT_PATCH_FIELD_ERRORS:
    logger.warning(
        "Collection blueprint and patch schema mismatch detected:\n%s",
        "\n".join(BLUEPRINT_PATCH_FIELD_ERRORS),
    )

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
REPLY_MODEL = os.environ.get("OPENAI_REPLY_MODEL", "gpt-4o-mini")
EXTRACTION_MODEL = os.environ.get("OPENAI_EXTRACTION_MODEL", "gpt-4o-mini")
WEB_PUSH_PUBLIC_KEY = os.environ.get("WEB_PUSH_PUBLIC_KEY", "").strip()
WEB_PUSH_PRIVATE_KEY = os.environ.get("WEB_PUSH_PRIVATE_KEY", "").strip()
WEB_PUSH_SUBJECT = os.environ.get("WEB_PUSH_SUBJECT", "mailto:admin@example.com").strip()

app = FastAPI(title="Crested Gecko Care")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


CHAT_STORE: dict[str, list[dict[str, str]]] = {}
PET_STORE: dict[str, PetProfile] = {}
PET_HISTORY_STORE: dict[str, PetCareHistory] = {}
CHAT_MESSAGE_STORE: dict[str, list[ChatMessage]] = {}
EXTRACTION_STORE: dict[str, list[ExtractionResult]] = {}
USER_STORE: dict[str, UserAccount] = {}
AI_CHARACTER_STORE: dict[str, AICharacterMemory] = {}
CARE_ANALYSIS_CACHE: dict[str, dict[str, Any]] = {}
CARE_REPORT_CACHE: dict[str, dict[str, Any]] = {}
CHAT_REPORT_PROMPT_STATE: dict[str, dict[str, Any]] = {}
RISK_NOTIFICATION_INBOX: dict[str, dict[str, Any]] = {}
RISK_NOTICE_CACHE: dict[str, dict[str, Any]] = {}
RISK_ALERT_SAME_SIGNATURE_COOLDOWN = timedelta(hours=12)
RISK_ALERT_LEVEL_COOLDOWN = {
    "alert": timedelta(hours=2),
    "watch": timedelta(hours=6),
}
RISK_ALERT_DAILY_CAP = {
    "alert": 4,
    "watch": 2,
}
RISK_ALERT_MIN_INTERVAL = timedelta(minutes=20)
KST = timezone(timedelta(hours=9))

TOPIC_LABELS = {
    "identity": "기본 정보",
    "feeding": "급여",
    "stool": "배변",
    "shed": "탈피",
    "behavior": "행동",
    "weight": "체중",
    "health": "건강",
    "habitat": "사육장",
    "routine": "루틴",
    "sensor": "센서",
    "sensor_settings": "센서",
}

FIELD_LABELS = {
    "identity.name": "이름",
    "identity.nickname": "별명",
    "feeding.primary_diet": "주식 종류",
    "feeding.appetite_level": "식욕 상태",
    "feeding.last_fed_at": "최근 급여 시점",
    "stool.last_stool_at": "최근 배변",
    "shed.last_shed_at": "최근 탈피",
    "weight.current_weight_grams": "현재 체중",
    "health.recent_concern_note": "최근 건강 메모",
    "habitat.day_temperature_c": "주간 온도",
    "habitat.day_humidity_percent": "주간 습도",
    "routine.feeding_time_window": "급여 시간대",
    "sensor_settings.sensor_enabled": "센서 사용 여부",
}


def profile_store_key(owner_user_id: str, pet_id: str) -> str:
    return f"{owner_user_id}::{pet_id}"


class SensorEnvironmentPayload(BaseModel):
    device_id: str = Field(min_length=1, max_length=120)
    temperature_c: float | None = Field(default=None, ge=0, le=50)
    humidity_percent: float | None = Field(default=None, ge=0, le=100)
    measured_at: datetime | None = None
    sensor_device_name: str | None = None


class SensorLinkPayload(BaseModel):
    device_id: str = Field(min_length=1, max_length=120)
    sensor_device_name: str | None = None
    sensor_location_note: str | None = None
    pet_name: str | None = None


class PushSubscriptionKeysPayload(BaseModel):
    p256dh: str = Field(min_length=1)
    auth: str = Field(min_length=1)


class PushSubscriptionPayload(BaseModel):
    endpoint: str = Field(min_length=1, max_length=2000)
    keys: PushSubscriptionKeysPayload
    expirationTime: int | None = None


class PushUnsubscribePayload(BaseModel):
    endpoint: str = Field(min_length=1, max_length=2000)


def clear_user_runtime_data(owner_user_id: str) -> None:
    USER_STORE.pop(owner_user_id, None)
    AI_CHARACTER_STORE.pop(owner_user_id, None)

    for key in list(PET_STORE.keys()):
        if key.startswith(f"{owner_user_id}::"):
            PET_STORE.pop(key, None)
    for key in list(PET_HISTORY_STORE.keys()):
        if key.startswith(f"{owner_user_id}::"):
            PET_HISTORY_STORE.pop(key, None)
    for key in list(CHAT_MESSAGE_STORE.keys()):
        if key.startswith(f"{owner_user_id}::"):
            CHAT_MESSAGE_STORE.pop(key, None)
    for key in list(EXTRACTION_STORE.keys()):
        if key.startswith(f"{owner_user_id}::"):
            EXTRACTION_STORE.pop(key, None)
    for key in list(CARE_ANALYSIS_CACHE.keys()):
        if key.startswith(f"{owner_user_id}::"):
            CARE_ANALYSIS_CACHE.pop(key, None)
    for key in list(CARE_REPORT_CACHE.keys()):
        if key.startswith(f"{owner_user_id}::"):
            CARE_REPORT_CACHE.pop(key, None)
    for key in list(CHAT_REPORT_PROMPT_STATE.keys()):
        if key.startswith(f"{owner_user_id}::"):
            CHAT_REPORT_PROMPT_STATE.pop(key, None)
    for key in list(RISK_NOTIFICATION_INBOX.keys()):
        if key.startswith(f"{owner_user_id}::"):
            RISK_NOTIFICATION_INBOX.pop(key, None)
    for key in list(RISK_NOTICE_CACHE.keys()):
        if key.startswith(f"{owner_user_id}::"):
            RISK_NOTICE_CACHE.pop(key, None)
    for session_key in list(CHAT_STORE.keys()):
        if session_key.startswith(f"chat_history::{owner_user_id}::"):
            CHAT_STORE.pop(session_key, None)


def delete_supabase_auth_user(user_id: str) -> tuple[bool, str]:
    supabase_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    service_role_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not supabase_url:
        return False, "SUPABASE_URL is missing"
    if not service_role_key:
        return False, "SUPABASE_SERVICE_ROLE_KEY is missing"

    try:
        req = UrlRequest(
            f"{supabase_url}/auth/v1/admin/users/{user_id}",
            headers={
                "apikey": service_role_key,
                "Authorization": f"Bearer {service_role_key}",
            },
            method="DELETE",
        )
        with urlopen(req, timeout=10):
            return True, "deleted"
    except HTTPError as exc:
        if exc.code == 404:
            return True, "already deleted"
        logger.warning("Supabase auth user deletion failed: %s", exc)
        return False, f"http_error_{exc.code}"
    except (URLError, TimeoutError) as exc:
        logger.warning("Supabase auth user deletion failed: %s", exc)
        return False, "network_error"


def get_request_user(request: Request) -> UserAccount:
    auth_user = resolve_auth_user(request)
    return get_or_create_user(
        user_id=auth_user["user_id"],
        email=auth_user.get("email"),
    )


def user_primary_pet_name(user: UserAccount) -> str | None:
    preferred = user.preferences.primary_pet_name
    if isinstance(preferred, CollectedField) and preferred.value:
        return resolve_pet_name(str(preferred.value))
    return None


def set_user_primary_pet_name(user: UserAccount, pet_name: str) -> str:
    resolved = resolve_pet_name(pet_name)
    user.preferences.primary_pet_name = CollectedField(
        value=resolved,
        confidence=ConfidenceLevel.HIGH,
        last_updated_at=utc_now(),
        needs_followup=False,
        followup_question_hint=None,
        evidence=[],
    )
    db_save_user(user.user_id, user.email, user)
    USER_STORE[user.user_id] = user
    return resolved


def resolve_pet_name_for_user(raw_name: str | None, user: UserAccount, fallback: str = "우리 도마뱀") -> str:
    if raw_name is not None and str(raw_name).strip():
        return resolve_pet_name(raw_name, fallback=fallback)
    preferred = user_primary_pet_name(user)
    if preferred:
        return preferred
    return fallback


def seed_character_memory(memory: AICharacterMemory) -> AICharacterMemory:
    if memory.daily_thoughts:
        return memory
    memory.daily_thoughts.append(
        AICharacterDailyThought(
            thought=(
                "나는 작은 변화나 패턴을 보는 걸 은근 좋아하는 편이야. "
                "온도나 습도 같은 것도 보면 괜히 흐름을 보고 싶어지더라."
            ),
            created_for_date=date.today(),
            tags=["seed", "persona"],
        )
    )
    memory.updated_at = utc_now()
    return memory


def get_or_create_ai_character_memory(user_id: str = "local-user") -> AICharacterMemory:
    memory = AI_CHARACTER_STORE.get(user_id)
    if memory is not None:
        memory = seed_character_memory_v2(memory)
        AI_CHARACTER_STORE[user_id] = memory
        db_save_character_memory(user_id, memory)
        return memory

    db_payload = db_load_character_memory(user_id)
    if db_payload:
        try:
            memory = AICharacterMemory.model_validate(db_payload)
        except Exception:
            logger.exception("Failed to validate AI character memory for user_id=%s", user_id)
            memory = AICharacterMemory()
    else:
        memory = AICharacterMemory()

    memory = seed_character_memory_v2(memory)
    AI_CHARACTER_STORE[user_id] = memory
    db_save_character_memory(user_id, memory)
    return memory


INITIAL_CHARACTER_MEMORY_COUNT = 20


def initial_character_seed_texts() -> list[str]:
    return [
        "오늘은 작은 변화가 왜 큰 힌트가 되는지 다시 생각했어.",
        "숫자로 기록된 하루에도 분위기가 있다는 걸 느꼈어.",
        "조용한 루틴이 쌓이면 신뢰가 된다는 말이 맞는 것 같아.",
        "급한 결론보다 한 번 더 관찰하는 태도가 더 멀리 가더라.",
        "익숙한 패턴이 깨지는 순간을 가장 먼저 기억해 두려고 해.",
        "먹이 반응은 속도보다 맥락이 더 중요하다는 걸 배웠어.",
        "같은 온도여도 컨디션에 따라 반응이 달라질 수 있더라.",
        "하루치 기록이 모이면 다음 선택이 훨씬 또렷해져.",
        "상태가 좋아 보이는 날일수록 기준선을 잘 남겨 두는 편이야.",
        "작은 거부 신호는 불편함의 시작일 때가 많아서 놓치지 않으려 해.",
        "습도 흐름은 값 하나보다 오르내림의 모양이 더 많은 걸 말해줘.",
        "관찰은 걱정을 키우기보다 불확실성을 줄여 주는 일 같아.",
        "변화가 없다는 정보도 의외로 중요한 단서가 되더라.",
        "리듬이 흔들릴 때는 원인을 하나씩 분리해 보는 게 도움이 돼.",
        "기록은 완벽함보다 꾸준함이 더 큰 힘을 낸다는 걸 믿어.",
        "몸 상태 힌트는 보통 여러 신호가 함께 움직일 때 선명해져.",
        "오늘은 평소와 같은지 묻는 질문이 꽤 좋은 시작점이었어.",
        "환경 조정은 크게 한 번보다 작게 여러 번이 안정적이더라.",
        "잘 먹는 날의 조건을 남겨 두면 다음에도 재현하기 쉬워져.",
        "불안할 때일수록 기본 루틴으로 돌아가는 게 가장 안전했어.",
        "컨디션 메모 한 줄이 나중에 큰 해석 차이를 만들기도 해.",
        "변화 원인을 찾을 땐 타이밍 정보를 꼭 함께 보려고 해.",
        "오늘의 정상 범위를 알아두면 이상 신호가 더 빨리 보이더라.",
        "과한 추정보다 지금 확인 가능한 사실부터 정리하는 쪽을 택해.",
        "결국 좋은 케어는 관찰과 대화가 같이 갈 때 완성되는 것 같아.",
    ]


def build_initial_character_thoughts(
    *,
    count: int = INITIAL_CHARACTER_MEMORY_COUNT,
    base_date: date | None = None,
) -> list[AICharacterDailyThought]:
    seed_texts = initial_character_seed_texts()
    target_count = min(count, len(seed_texts))
    if target_count <= 0:
        return []

    anchor = base_date or date.today()
    start_date = anchor - timedelta(days=target_count)
    thoughts: list[AICharacterDailyThought] = []
    for offset, text in enumerate(seed_texts[:target_count]):
        tags = ["seed", "persona"] if offset == 0 else ["seed", "daily", "story"]
        thoughts.append(
            AICharacterDailyThought(
                thought=text,
                created_for_date=start_date + timedelta(days=offset),
                tags=tags,
            )
        )
    return thoughts


def seed_character_memory_v2(memory: AICharacterMemory) -> AICharacterMemory:
    memory.max_daily_thoughts = max(memory.max_daily_thoughts, INITIAL_CHARACTER_MEMORY_COUNT)
    if len(memory.daily_thoughts) >= INITIAL_CHARACTER_MEMORY_COUNT:
        return memory

    seed_thoughts = build_initial_character_thoughts(count=INITIAL_CHARACTER_MEMORY_COUNT)
    existing_signatures = {
        (item.thought.strip(), item.created_for_date.isoformat())
        for item in memory.daily_thoughts
    }
    for item in seed_thoughts:
        signature = (item.thought.strip(), item.created_for_date.isoformat())
        if signature in existing_signatures:
            continue
        memory.daily_thoughts.append(item)
        existing_signatures.add(signature)
        if len(memory.daily_thoughts) >= INITIAL_CHARACTER_MEMORY_COUNT:
            break

    memory.daily_thoughts = memory.daily_thoughts[-memory.max_daily_thoughts:]
    memory.updated_at = utc_now()
    return memory


def latest_user_text(history: list[dict[str, str]]) -> str:
    for item in reversed(history):
        if item.get("role") == "user":
            return item.get("text", "")
    return ""


def is_short_closing_message(message_text: str) -> bool:
    cleaned = message_text.strip().lower()
    if not cleaned:
        return True
    closing_words = {
        "응", "ㅇㅇ", "아니", "아냐", "없어", "괜찮아", "고마워", "감사", "그래", "오케이",
        "ok", "okay", "no", "nope", "thanks", "thank you",
    }
    return len(cleaned) <= 8 and any(word in cleaned for word in closing_words)


def user_message_is_question(message_text: str) -> bool:
    cleaned = message_text.strip()
    question_markers = ("?", "뭐", "왜", "어떻게", "언제", "가능", "되나", "돼", "인가", "일까", "맞아")
    return any(marker in cleaned for marker in question_markers)


def is_serious_care_context(message_text: str, followup_plan: dict[str, Any]) -> bool:
    if followup_plan.get("conversation_state") == ConversationState.CONCERN.value:
        return True
    lowered = message_text.lower()
    serious_keywords = (
        "아파", "죽", "피", "상처", "호흡", "무기력", "탈수", "병원", "응급",
        "sick", "injury", "blood", "emergency",
    )
    return any(keyword in lowered for keyword in serious_keywords)


def choose_character_thought(memory: AICharacterMemory) -> str | None:
    if not memory.daily_thoughts:
        return None
    return memory.daily_thoughts[-1].thought


def should_use_character_line(
    history: list[dict[str, str]],
    followup_plan: dict[str, Any],
    memory: AICharacterMemory | None,
) -> bool:
    if memory is None or not memory.daily_thoughts:
        return False
    if followup_plan.get("should_ask"):
        return False

    message_text = latest_user_text(history)
    if is_short_closing_message(message_text):
        return False
    if user_message_is_question(message_text):
        return False
    if is_serious_care_context(message_text, followup_plan):
        return False

    # 자기 이야기는 양념처럼만 사용합니다. 대략 25% 확률로만 허용합니다.
    return random.random() < 0.25


def character_memory_prompt_context(
    memory: AICharacterMemory | None,
    history: list[dict[str, str]],
    followup_plan: dict[str, Any],
) -> dict[str, Any]:
    if memory is None:
        return {"enabled": False}

    enabled = should_use_character_line(history, followup_plan, memory)
    persona = memory.persona
    return {
        "enabled": enabled,
        "name": persona.name,
        "personality": persona.personality,
        "likes": persona.likes[:4],
        "speaking_style": persona.speaking_style,
        "daily_thought": choose_character_thought(memory) if enabled else None,
        "usage_rule": (
            "enabled가 true일 때도 반드시 사용하지는 말고, 대화 흐름에 자연스러울 때만 한 줄 이하로 섞는다. "
            "실제 인간 경험처럼 말하지 말고 취향, 생각, 비유처럼 표현한다."
        ),
    }


async def refresh_ai_character_memory_if_needed(memory: AICharacterMemory, owner_user_id: str = "local-user") -> None:
    today = date.today()
    if any(item.created_for_date == today for item in memory.daily_thoughts):
        return

    fallback_thoughts = [
        "오늘은 작은 변화도 기록으로 남기면 나중에 꽤 큰 단서가 된다는 생각이 들었어.",
        "온도나 습도처럼 숫자로 보이는 것들은 가끔 조용한 일기 같다고 느껴져.",
        "나는 패턴을 보는 걸 좋아해서 그런지, 사소한 루틴 변화도 은근 눈에 들어오는 편이야.",
    ]

    thought: str | None = None
    if OPENAI_API_KEY:
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            prompt = (
                "너는 크레스티드 게코 케어를 돕는 AI 친구의 캐릭터 메모리를 만든다.\n"
                "오늘의 짧은 생각을 1문장으로 작성해라.\n"
                "규칙:\n"
                "- 실제 인간 경험처럼 말하지 않는다.\n"
                "- 취향, 관찰, 비유, 생각처럼 표현한다.\n"
                "- 과하게 감성적이지 않게, 가볍고 자연스럽게 쓴다.\n"
                "- 한국어로 쓴다.\n"
                "JSON 형식으로만 출력: {\"thought\": \"...\"}"
            )
            completion = await client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}],
                model=REPLY_MODEL,
                temperature=0.8,
                max_completion_tokens=120,
                response_format={"type": "json_object"},
            )
            payload = json.loads(completion.choices[0].message.content or "{}")
            raw_thought = str(payload.get("thought", "")).strip()
            if raw_thought:
                thought = raw_thought
        except Exception as e:
            logger.exception("AI character memory generation failed: %s", e)

    if thought is None:
        thought = fallback_thoughts[today.toordinal() % len(fallback_thoughts)]

    memory.daily_thoughts.append(
        AICharacterDailyThought(
            thought=thought,
            created_for_date=today,
            tags=["daily", "story"],
        )
    )
    memory.daily_thoughts = memory.daily_thoughts[-memory.max_daily_thoughts:]
    memory.updated_at = utc_now()
    AI_CHARACTER_STORE[owner_user_id] = memory
    db_save_character_memory(owner_user_id, memory)


def resolve_pet_name(raw_name: str | None, fallback: str = "우리 도마뱀") -> str:
    if raw_name is None:
        return fallback
    cleaned = raw_name.strip()
    return cleaned or fallback


def build_path(path: str, **params: str | None) -> str:
    filtered = {key: value for key, value in params.items() if value}
    if not filtered:
        return path
    return f"{path}?{urlencode(filtered)}"


def build_install_url() -> str:
    return "/main"


def web_push_enabled() -> bool:
    return bool(WEB_PUSH_PUBLIC_KEY and WEB_PUSH_PRIVATE_KEY and WEB_PUSH_SUBJECT)


def upsert_push_subscription(
    user: UserAccount,
    payload: PushSubscriptionPayload,
    user_agent: str | None = None,
) -> None:
    now = utc_now()
    subscriptions = list(user.preferences.push_subscriptions or [])
    updated: list[WebPushSubscription] = []
    replaced = False

    for item in subscriptions:
        if item.endpoint == payload.endpoint:
            updated.append(
                WebPushSubscription(
                    endpoint=payload.endpoint,
                    keys=WebPushSubscriptionKeys(
                        p256dh=payload.keys.p256dh,
                        auth=payload.keys.auth,
                    ),
                    expirationTime=payload.expirationTime,
                    user_agent=user_agent or item.user_agent,
                    created_at=item.created_at,
                    updated_at=now,
                )
            )
            replaced = True
        else:
            updated.append(item)

    if not replaced:
        updated.append(
            WebPushSubscription(
                endpoint=payload.endpoint,
                keys=WebPushSubscriptionKeys(
                    p256dh=payload.keys.p256dh,
                    auth=payload.keys.auth,
                ),
                expirationTime=payload.expirationTime,
                user_agent=user_agent,
                created_at=now,
                updated_at=now,
            )
        )

    user.preferences.push_subscriptions = updated[-5:]
    USER_STORE[user.user_id] = user
    db_save_user(user.user_id, user.email, user)


def remove_push_subscription(user: UserAccount, endpoint: str) -> bool:
    subscriptions = list(user.preferences.push_subscriptions or [])
    filtered = [item for item in subscriptions if item.endpoint != endpoint]
    changed = len(filtered) != len(subscriptions)
    if changed:
        user.preferences.push_subscriptions = filtered
        USER_STORE[user.user_id] = user
        db_save_user(user.user_id, user.email, user)
    return changed


def send_web_push_notifications(user: UserAccount, payload: dict[str, Any]) -> dict[str, int]:
    subscriptions = list(user.preferences.push_subscriptions or [])
    if not subscriptions:
        return {"attempted": 0, "sent": 0}
    if not web_push_enabled():
        return {"attempted": len(subscriptions), "sent": 0}

    try:
        from pywebpush import WebPushException, webpush
    except Exception:
        logger.warning("pywebpush is not installed; push notifications are disabled.")
        return {"attempted": len(subscriptions), "sent": 0}

    sent = 0
    stale_endpoints: list[str] = []
    data = json.dumps(payload, ensure_ascii=False)
    for sub in subscriptions:
        try:
            webpush(
                subscription_info={
                    "endpoint": sub.endpoint,
                    "keys": {
                        "p256dh": sub.keys.p256dh,
                        "auth": sub.keys.auth,
                    },
                },
                data=data,
                vapid_private_key=WEB_PUSH_PRIVATE_KEY,
                vapid_claims={"sub": WEB_PUSH_SUBJECT},
                ttl=120,
            )
            sent += 1
        except WebPushException as exc:
            status_code = None
            if getattr(exc, "response", None) is not None:
                status_code = getattr(exc.response, "status_code", None)
            if status_code in {404, 410}:
                stale_endpoints.append(sub.endpoint)
            logger.info("Web push delivery failed for user=%s status=%s", user.user_id, status_code)
        except Exception:
            logger.exception("Unexpected web push failure for user=%s", user.user_id)

    for endpoint in stale_endpoints:
        remove_push_subscription(user, endpoint)
    return {"attempted": len(subscriptions), "sent": sent}


def get_chat_session_key(pet_name: str, owner_user_id: str = "local-user") -> str:
    return f"chat_history::{owner_user_id}::{pet_name}"


def slugify_pet_id(pet_name: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in pet_name.strip())
    compact = "_".join(part for part in normalized.split("_") if part)
    return compact or "crested_gecko"


def get_or_create_user(user_id: str = "local-user", email: str | None = None) -> UserAccount:
    user = USER_STORE.get(user_id)
    if user is not None:
        if email and user.email != email:
            user.email = email
            db_save_user(user_id, email, user)
        return user

    db_payload = db_load_user(user_id)
    if db_payload and db_payload.get("data"):
        try:
            user = UserAccount.model_validate(db_payload["data"])
        except Exception:
            logger.exception("Failed to validate user row for user_id=%s", user_id)
            user = UserAccount(user_id=user_id, email=email or db_payload.get("email"))
    else:
        user = UserAccount(user_id=user_id, email=email)

    if email and user.email != email:
        user.email = email

    USER_STORE[user_id] = user
    db_save_user(user_id, user.email, user)
    return user


def get_or_create_pet_profile(pet_name: str, owner_user_id: str = "local-user") -> PetProfile:
    pet_id = slugify_pet_id(pet_name)
    cache_key = profile_store_key(owner_user_id, pet_id)
    profile = PET_STORE.get(cache_key)
    if profile is not None:
        return profile

    db_payload = db_load_pet_profile(owner_user_id, pet_id)
    if db_payload:
        try:
            profile = PetProfile.model_validate(db_payload)
        except Exception:
            logger.exception("Failed to validate pet profile for pet_id=%s", pet_id)
            profile = None
    else:
        profile = None

    if profile is None:
        profile = PetProfile(
            pet_id=pet_id,
            owner_user_id=owner_user_id,
        )
        profile.identity.name = CollectedField(
            value=pet_name,
            confidence=ConfidenceLevel.HIGH,
            last_updated_at=utc_now(),
        )

    PET_STORE[cache_key] = profile
    db_save_pet_profile(owner_user_id, pet_id, profile)
    return profile


def get_or_create_pet_profile_by_id(
    pet_id: str,
    owner_user_id: str = "local-user",
    pet_name_hint: str | None = None,
) -> PetProfile:
    normalized_pet_id = str(pet_id).strip()
    if not normalized_pet_id:
        raise ValueError("pet_id is required")

    cache_key = profile_store_key(owner_user_id, normalized_pet_id)
    profile = PET_STORE.get(cache_key)
    if profile is not None:
        return profile

    db_payload = db_load_pet_profile(owner_user_id, normalized_pet_id)
    if db_payload:
        try:
            profile = PetProfile.model_validate(db_payload)
        except Exception:
            logger.exception("Failed to validate pet profile for pet_id=%s", normalized_pet_id)
            profile = None
    else:
        profile = None

    if profile is None:
        fallback_name = resolve_pet_name(pet_name_hint, fallback=normalized_pet_id)
        profile = PetProfile(
            pet_id=normalized_pet_id,
            owner_user_id=owner_user_id,
        )
        profile.identity.name = CollectedField(
            value=fallback_name,
            confidence=ConfidenceLevel.LOW,
            last_updated_at=utc_now(),
        )

    PET_STORE[cache_key] = profile
    db_save_pet_profile(owner_user_id, normalized_pet_id, profile)
    return profile


def find_sensor_target_profile(
    device_id: str,
) -> tuple[str, PetProfile] | None:
    normalized_device_id = str(device_id).strip()
    if not normalized_device_id:
        return None

    for cache_key, profile in PET_STORE.items():
        configured_device_id = str(collected_value(profile, "sensor_settings.sensor_device_id") or "").strip()
        if not configured_device_id:
            continue
        if configured_device_id != normalized_device_id:
            continue
        owner_user_id, _, _ = cache_key.partition("::")
        return owner_user_id or profile.owner_user_id, profile

    row = db_load_sensor_link_target_by_device_id(normalized_device_id)
    if not row:
        return None

    owner_user_id = str(row.get("owner_user_id") or "").strip()
    pet_id = str(row.get("pet_id") or "").strip()
    payload = row.get("data")
    if not owner_user_id or not pet_id or not payload:
        return None

    try:
        profile = PetProfile.model_validate(payload)
    except Exception:
        logger.exception("Failed to validate sensor-linked pet profile for pet_id=%s", pet_id)
        return None

    PET_STORE[profile_store_key(owner_user_id, pet_id)] = profile
    return owner_user_id, profile


def set_collected_field_value(profile: PetProfile, field_path: str, value: Any, confidence: ConfidenceLevel = ConfidenceLevel.HIGH) -> None:
    parts = field_path.split(".")
    cursor: Any = profile
    for part in parts[:-1]:
        if not hasattr(cursor, part):
            return
        cursor = getattr(cursor, part)
    leaf = parts[-1]
    if not hasattr(cursor, leaf):
        return
    current = getattr(cursor, leaf)
    existing_evidence = current.evidence if isinstance(current, CollectedField) else []
    setattr(
        cursor,
        leaf,
        CollectedField(
            value=value,
            confidence=confidence,
            last_updated_at=utc_now(),
            needs_followup=False,
            followup_question_hint=None,
            evidence=existing_evidence,
        ),
    )


def get_or_create_pet_history(pet_id: str, owner_user_id: str = "local-user") -> PetCareHistory:
    cache_key = profile_store_key(owner_user_id, pet_id)
    history = PET_HISTORY_STORE.get(cache_key)
    if history is not None:
        return history

    db_payload = db_load_pet_history(owner_user_id, pet_id)
    if db_payload:
        try:
            history = PetCareHistory.model_validate(db_payload)
        except Exception:
            logger.exception("Failed to validate pet history for pet_id=%s", pet_id)
            history = None
    else:
        history = None

    if history is None:
        history = PetCareHistory(pet_id=pet_id)

    PET_HISTORY_STORE[cache_key] = history
    db_save_pet_history(owner_user_id, pet_id, history)
    return history


def profile_summary(profile: PetProfile) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "pet_id": profile.pet_id,
        "species": profile.species,
    }
    for section_name in (
        "identity",
        "feeding",
        "stool",
        "shed",
        "behavior",
        "weight",
        "health",
        "habitat",
        "routine",
        "sensor_settings",
    ):
        section = getattr(profile, section_name)
        section_values: dict[str, Any] = {}
        for field_name in section.__class__.model_fields:
            field_value = getattr(section, field_name)
            if isinstance(field_value, CollectedField) and field_value.value is not None:
                section_values[field_name] = field_value.value
            elif hasattr(field_value, "model_dump"):
                dumped = field_value.model_dump(mode="json", exclude_none=True)
                if dumped:
                    section_values[field_name] = dumped
            elif isinstance(field_value, list) and field_value:
                section_values[field_name] = [
                    item.model_dump(mode="json", exclude_none=True)
                    if hasattr(item, "model_dump")
                    else item
                    for item in field_value
                ]
        if section_values:
            summary[section_name] = section_values
    return summary


def history_summary(history: PetCareHistory, limit: int = 5) -> dict[str, Any]:
    return {
        "pet_id": history.pet_id,
        "counts": {
            "weight_logs": len(history.weight_logs),
            "feeding_logs": len(history.feeding_logs),
            "environment_readings": len(history.environment_readings),
            "environment_daily_summaries": len(history.environment_daily_summaries),
            "care_events": len(history.care_events),
            "photo_records": len(history.photo_records),
            "monthly_summaries": len(history.monthly_summaries),
        },
        "recent_weight_logs": [
            item.model_dump(mode="json", exclude_none=True)
            for item in history.weight_logs[-limit:]
        ],
        "recent_feeding_logs": [
            item.model_dump(mode="json", exclude_none=True)
            for item in history.feeding_logs[-limit:]
        ],
        "recent_environment_readings": [
            item.model_dump(mode="json", exclude_none=True)
            for item in history.environment_readings[-limit:]
        ],
        "latest_environment_daily_summary": (
            history.environment_daily_summaries[-1].model_dump(mode="json", exclude_none=True)
            if history.environment_daily_summaries
            else None
        ),
        "recent_care_events": [
            item.model_dump(mode="json", exclude_none=True)
            for item in history.care_events[-limit:]
        ],
        "retention_policy": history.retention_policy.model_dump(mode="json"),
    }


def to_kst(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(KST)


def format_dashboard_date(moment: datetime) -> str:
    weekday_labels = ["월", "화", "수", "목", "금", "토", "일"]
    local_moment = to_kst(moment)
    return f"{local_moment.month}월 {local_moment.day}일 {weekday_labels[local_moment.weekday()]}요일"


def topic_label(raw_topic: str | None) -> str:
    if not raw_topic:
        return "없음"
    return TOPIC_LABELS.get(raw_topic, raw_topic.replace("_", " "))


def field_label(field_path: str) -> str:
    if field_path in FIELD_LABELS:
        return FIELD_LABELS[field_path]
    tail = field_path.split(".")[-1]
    return tail.replace("_", " ")


FRIENDLY_WORD_REPLACEMENTS = {
    "급여": "밥",
    "배변": "응가",
    "탈피": "허물 벗기",
    "섭취": "먹은 양",
    "기록": "메모",
    "측정": "재기",
    "확인": "체크",
    "이상 징후": "이상 신호",
    "권장": "추천",
    "조치": "해보면 좋은 것",
    "상태": "컨디션",
}


def to_friendly_question_text(text: str) -> str:
    friendly = str(text or "").strip()
    if not friendly:
        return friendly

    for src, dst in FRIENDLY_WORD_REPLACEMENTS.items():
        friendly = friendly.replace(src, dst)

    friendly = friendly.replace("알려주실래요?", "알려줄래?")
    friendly = friendly.replace("알려 주세요.", "알려줘.")
    friendly = friendly.replace("주세요.", "줘.")
    friendly = friendly.replace("해 주세요.", "해줘.")
    friendly = friendly.replace("있었는지", "있었는지")

    if "?" not in friendly and "요?" not in friendly:
        friendly = friendly.rstrip(".") + "?"
    return friendly


def friendly_topic_label(raw_topic: str | None) -> str:
    topic = topic_label(raw_topic)
    return to_friendly_question_text(topic).rstrip("?")


def friendly_field_label(field_path: str) -> str:
    label = field_label(field_path)
    return to_friendly_question_text(label).rstrip("?")


BASIC_GECKO_ONBOARDING_HINTS: list[tuple[str, str]] = [
    ("identity.name", "우리 도마뱀 이름이 뭐야?"),
    ("identity.nickname", "평소에 뭐라고 불러?"),
    ("feeding.primary_diet", "요즘 주로 뭐 먹고 있어?"),
    ("feeding.appetite_level", "요즘 밥 먹는 기운은 어때?"),
    ("habitat.day_temperature_c", "낮 온도는 보통 몇 도쯤으로 맞춰?"),
    ("habitat.day_humidity_percent", "습도는 보통 몇 퍼센트쯤이야?"),
    ("weight.current_weight_grams", "최근 몸무게 재봤으면 몇 g 정도야?"),
    ("stool.last_stool_at", "최근 응가 본 건 언제쯤이야?"),
    ("shed.last_shed_at", "최근 허물 벗기는 언제였어?"),
]


def first_basic_onboarding_hint(profile: PetProfile) -> dict[str, str] | None:
    for field_path, question in BASIC_GECKO_ONBOARDING_HINTS:
        if not field_needs_followup(profile, field_path):
            continue
        topic = field_path.split(".")[0]
        return {
            "topic": topic,
            "field_path": field_path,
            "question_hint": to_friendly_question_text(question),
            "reason": "basic gecko onboarding hint",
        }
    return None


def format_timestamp_label(value: datetime | None) -> str:
    if value is None:
        return "방금 전"
    return to_kst(value).strftime("%m.%d %H:%M")


def build_dashboard_tasks(profile: PetProfile) -> list[dict[str, str]]:
    missing_fields = profile.coverage.missing_fields[:4]
    if missing_fields:
        return [
            {
                "label": field_label(item),
                "status": topic_label(item.split(".")[0] if "." in item else item),
            }
            for item in missing_fields
        ]

    next_topic = topic_label(profile.coverage.next_best_question_topic)
    return [
        {"label": "오늘 급여 반응 확인", "status": "추천"},
        {"label": "온도와 습도 기록", "status": "루틴"},
        {"label": f"다음 체크 주제: {next_topic}", "status": "AI"},
    ]


def build_activity_feed(history: PetCareHistory) -> list[dict[str, str]]:
    feed: list[tuple[datetime, dict[str, str]]] = []

    for item in history.weight_logs[-2:]:
        feed.append(
            (
                item.measured_at,
                {
                    "category": "체중",
                    "title": f"{item.weight_grams}g 측정",
                    "detail": item.body_condition_note or "체형 메모 없이 저장됨",
                    "meta": format_timestamp_label(item.measured_at),
                    "tone": "weight",
                },
            )
        )

    for item in history.feeding_logs[-2:]:
        food_name = item.food_name or item.food_category.value
        feed.append(
            (
                item.offered_at,
                {
                    "category": "급여",
                    "title": f"{food_name} / {item.outcome.value}",
                    "detail": item.amount_eaten_note or item.refusal_reason_note or "급여 로그가 저장됐어요.",
                    "meta": format_timestamp_label(item.offered_at),
                    "tone": "feeding",
                },
            )
        )

    for item in history.environment_readings[-2:]:
        temperature_text = (
            f"{item.temperature_c:.1f}°C" if item.temperature_c is not None else "온도 없음"
        )
        humidity_text = (
            f"{item.humidity_percent:.0f}%" if item.humidity_percent is not None else "습도 없음"
        )
        feed.append(
            (
                item.measured_at,
                {
                    "category": "환경",
                    "title": f"{temperature_text} / {humidity_text}",
                    "detail": item.sensor_device_name or "채팅에서 저장된 환경 기록",
                    "meta": format_timestamp_label(item.measured_at),
                    "tone": "habitat",
                },
            )
        )

    for item in history.care_events[-3:]:
        feed.append(
            (
                item.occurred_at,
                {
                    "category": "메모",
                    "title": item.title or item.event_type.value,
                    "detail": item.note or "저장된 메모가 있어요.",
                    "meta": format_timestamp_label(item.occurred_at),
                    "tone": "note",
                },
            )
        )

    feed.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in feed[:5]]


def build_environment_chart_points(history: PetCareHistory, limit: int = 24) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for item in history.environment_readings:
        if item.temperature_c is None and item.humidity_percent is None:
            continue
        points.append(
            {
                "label": to_kst(item.measured_at).strftime("%m-%d %H:%M"),
                "temperature": float(item.temperature_c) if item.temperature_c is not None else None,
                "humidity": float(item.humidity_percent) if item.humidity_percent is not None else None,
            }
        )
    return points[-limit:]


def build_profile_lines(profile: PetProfile) -> list[str]:
    facts: list[str] = []

    primary_diet = collected_value(profile, "feeding.primary_diet")
    if primary_diet:
        facts.append(f"기본 주식: {primary_diet}")

    appetite = collected_value(profile, "feeding.appetite_level")
    if appetite:
        appetite_text = appetite.value if hasattr(appetite, "value") else str(appetite)
        facts.append(f"식욕 상태: {appetite_text}")

    weight = collected_value(profile, "weight.current_weight_grams")
    if weight is not None:
        facts.append(f"최근 체중: {weight}g")

    day_temp = collected_value(profile, "habitat.day_temperature_c")
    day_humidity = collected_value(profile, "habitat.day_humidity_percent")
    if day_temp is not None or day_humidity is not None:
        temp_text = f"{day_temp}°C" if day_temp is not None else "-"
        humidity_text = f"{day_humidity}%" if day_humidity is not None else "-"
        facts.append(f"주간 환경: {temp_text} / {humidity_text}")

    if not facts:
        facts.append("아직 프로필이 많이 비어 있어요. 채팅에서 한두 문장만 남겨도 바로 채워집니다.")

    return facts[:4]


def _merge_care_levels(*levels: str) -> str:
    ranking = {"stable": 0, "watch": 1, "alert": 2}
    merged = "stable"
    for level in levels:
        if ranking.get(level, 0) > ranking.get(merged, 0):
            merged = level
    return merged


def _care_level_label(level: str) -> str:
    labels = {
        "stable": "안정",
        "watch": "주의",
        "alert": "경고",
    }
    return labels.get(level, "안정")


def _safe_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.0f}%"


def refresh_care_snapshots(profile: PetProfile, history: PetCareHistory) -> None:
    if history.weight_logs:
        sync_weight_snapshot(profile, history)
    if history.feeding_logs:
        sync_feeding_snapshot(profile, history)
    if history.environment_readings:
        sync_environment_snapshot(profile, history)


def build_rule_care_analysis(profile: PetProfile, history: PetCareHistory) -> dict[str, Any]:
    refresh_care_snapshots(profile, history)
    now = utc_now()
    rule_signals: list[dict[str, str]] = []
    recommended_actions: list[str] = []

    feeding_stats = profile.feeding.recent_stats
    feeding_level = "stable"
    feeding_text = "최근 급여 기록이 부족합니다."
    if feeding_stats and feeding_stats.offered_count > 0:
        refusal_rate = (
            feeding_stats.refused_count / feeding_stats.offered_count
            if feeding_stats.offered_count
            else 0.0
        )
        if feeding_stats.refusal_streak_days >= 2 or (
            feeding_stats.offered_count >= 3 and refusal_rate >= 0.6
        ):
            feeding_level = "alert"
            feeding_text = "급여 거부가 연속으로 이어지고 있습니다."
            recommended_actions.append("사육장을 안정시키고 24시간 내 급여를 다시 시도하세요.")
            recommended_actions.append("저활동과 거부가 지속되면 파충류 진료 가능한 병원 상담을 권장합니다.")
        elif feeding_stats.offered_count >= 3 and refusal_rate >= 0.35:
            feeding_level = "watch"
            feeding_text = "이번 주 급여 반응이 불안정합니다."
            recommended_actions.append("급여 시간을 고정하고 섭취량을 함께 기록하세요.")
        else:
            feeding_text = "급여 반응은 대체로 안정적입니다."

    rule_signals.append(
        {
            "area": "급여",
            "level": feeding_level,
            "level_label": _care_level_label(feeding_level),
            "summary": feeding_text,
        }
    )

    weight_trend = profile.weight.recent_trend
    weight_level = "stable"
    weight_text = "체중 추세 데이터가 아직 부족합니다."
    if weight_trend and weight_trend.change_percent is not None:
        delta = weight_trend.change_percent
        if delta <= -7:
            weight_level = "alert"
            weight_text = f"체중이 급격히 감소했습니다 ({delta:.1f}%)."
            recommended_actions.append("같은 저울/시간대로 24~48시간 내 재측정하세요.")
        elif delta <= -3:
            weight_level = "watch"
            weight_text = f"체중이 감소 추세입니다 ({delta:.1f}%)."
            recommended_actions.append("2~3일간 식욕과 배변을 함께 관찰해 기록하세요.")
        elif delta >= 8:
            weight_level = "watch"
            weight_text = f"체중 변화 폭이 큽니다 (+{delta:.1f}%)."
        else:
            weight_text = f"체중 추세는 안정적입니다 ({delta:+.1f}%)."

    rule_signals.append(
        {
            "area": "체중",
            "level": weight_level,
            "level_label": _care_level_label(weight_level),
            "summary": weight_text,
        }
    )

    env_risk = profile.habitat.recent_environment_risk
    habitat_level = "stable"
    habitat_text = "최근 24시간 환경 측정값이 부족합니다."
    if env_risk:
        alerts = {item.value if hasattr(item, "value") else str(item) for item in env_risk.alert_types}
        has_temp_critical = "overheat" in alerts or "too_cold" in alerts
        has_humidity_alert = "low_humidity" in alerts or "high_humidity" in alerts
        if has_temp_critical:
            habitat_level = "alert"
            habitat_text = "최근 환경 기록에서 온도 위험 신호가 감지되었습니다."
            recommended_actions.append("가열/냉각 장치를 점검하고 현재 사육장 온도를 확인하세요.")
        elif has_humidity_alert:
            habitat_level = "watch"
            habitat_text = "최근 습도가 권장 범위를 벗어났습니다."
            recommended_actions.append("분무/환기 설정을 조정하고 오늘 안에 습도를 재확인하세요.")
        else:
            habitat_text = "환경 추세는 현재 안정적입니다."

    rule_signals.append(
        {
            "area": "환경",
            "level": habitat_level,
            "level_label": _care_level_label(habitat_level),
            "summary": habitat_text,
        }
    )

    health_level = "stable"
    health_text = "최근 고우선 건강 이벤트는 없습니다."
    recent_events = [event for event in history.care_events if event.occurred_at >= now - timedelta(days=7)]
    high_health_event = any(
        event.event_type in {CareEventType.HEALTH, CareEventType.VET_VISIT, CareEventType.MEDICATION}
        and (
            event.importance == CareEventImportance.IMPORTANT
            or event.symptom_severity in {SymptomSeverity.HIGH, SymptomSeverity.URGENT}
        )
        for event in recent_events
    )
    watch_health_event = any(
        event.event_type in {CareEventType.HEALTH, CareEventType.STOOL, CareEventType.SHED}
        and event.importance in {CareEventImportance.NOTEWORTHY, CareEventImportance.IMPORTANT}
        for event in recent_events
    )
    if high_health_event:
        health_level = "alert"
        health_text = "최근 건강 이벤트의 밀접 관찰이 필요합니다."
        recommended_actions.append("행동/수분상태를 집중 관찰하고 증상 지속 시 진료 요약을 준비하세요.")
    elif watch_health_event:
        health_level = "watch"
        health_text = "최근 주의가 필요한 케어 이벤트가 있습니다."

    rule_signals.append(
        {
            "area": "건강",
            "level": health_level,
            "level_label": _care_level_label(health_level),
            "summary": health_text,
        }
    )

    overall_level = _merge_care_levels(feeding_level, weight_level, habitat_level, health_level)

    if not recommended_actions:
        recommended_actions = [
            "루틴 점검을 유지하고 급여·체중·환경 기록을 꾸준히 남겨주세요.",
            "작은 행동 변화도 채팅에 남기면 분석 정확도가 올라갑니다.",
        ]

    feeding_metric = "-"
    if feeding_stats and feeding_stats.offered_count > 0:
        feeding_metric = _safe_percent(feeding_stats.acceptance_rate)

    weight_metric = "-"
    if weight_trend and weight_trend.change_percent is not None:
        weight_metric = f"{weight_trend.change_percent:+.1f}%"

    env_metric = "0"
    if env_risk:
        env_metric = str(len(env_risk.alert_types))

    return {
        "overall_level": overall_level,
        "overall_label": _care_level_label(overall_level),
        "rule_signals": rule_signals,
        "recommended_actions": recommended_actions[:3],
        "metrics": [
            {"label": "급여", "value": feeding_metric, "note": "7일 수용률"},
            {"label": "체중", "value": weight_metric, "note": "최근 변화"},
            {"label": "환경", "value": env_metric, "note": "24시간 경보"},
        ],
    }


def care_analysis_signature(history: PetCareHistory) -> str:
    def last_timestamp(items: list[Any], attribute: str) -> str:
        if not items:
            return "-"
        value = getattr(items[-1], attribute, None)
        if isinstance(value, datetime):
            return value.isoformat()
        return "-"

    return "|".join(
        [
            str(len(history.weight_logs)),
            last_timestamp(history.weight_logs, "measured_at"),
            str(len(history.feeding_logs)),
            last_timestamp(history.feeding_logs, "offered_at"),
            str(len(history.environment_readings)),
            last_timestamp(history.environment_readings, "measured_at"),
            str(len(history.care_events)),
            last_timestamp(history.care_events, "occurred_at"),
        ]
    )


def fallback_ai_care_summary(rule_analysis: dict[str, Any]) -> dict[str, Any]:
    level = rule_analysis.get("overall_level", "stable")
    if level == "alert":
        summary = "복수의 위험 신호가 감지되었습니다. 오늘은 안정화 점검을 우선하세요."
    elif level == "watch":
        summary = "일부 지표가 불안정합니다. 24시간 내 집중 추적을 권장합니다."
    else:
        summary = "현재 지표는 안정적으로 보입니다. 루틴 케어와 기록을 유지하세요."

    return {
        "situation": "룰 기반 신호를 바탕으로 AI가 맥락 해석을 보강한 결과입니다.",
        "status_summary": summary,
        "priority_actions": rule_analysis.get("recommended_actions", [])[:3],
    }


async def generate_ai_care_summary(
    profile: PetProfile,
    history: PetCareHistory,
    rule_analysis: dict[str, Any],
) -> dict[str, Any]:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        return fallback_ai_care_summary(rule_analysis)

    if not OPENAI_API_KEY:
        return fallback_ai_care_summary(rule_analysis)

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    prompt = (
        "You are a reptile care assistant. You are given rule-based care signals.\n"
        "Interpret context and prioritize practical care actions.\n"
        "Do not diagnose disease and do not overstate certainty.\n"
        "All output must be written in Korean.\n"
        "Return JSON only with keys: situation, status_summary, priority_actions.\n"
        "priority_actions must be a short list of max 3 actionable items.\n"
        f"profile={json.dumps(compact_profile_context(profile), ensure_ascii=False, default=str)}\n"
        f"history={json.dumps(compact_history_context(history), ensure_ascii=False, default=str)}\n"
        f"rule_analysis={json.dumps(rule_analysis, ensure_ascii=False, default=str)}"
    )

    try:
        completion = await client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            model=REPLY_MODEL,
            temperature=0.2,
            max_completion_tokens=220,
            response_format={"type": "json_object"},
        )
        payload = json.loads(completion.choices[0].message.content or "{}")
        actions = payload.get("priority_actions")
        if not isinstance(actions, list):
            actions = rule_analysis.get("recommended_actions", [])
        normalized_actions = [str(item).strip() for item in actions if str(item).strip()][:3]
        if not normalized_actions:
            normalized_actions = rule_analysis.get("recommended_actions", [])[:3]

        return {
            "situation": str(payload.get("situation", "")).strip() or "AI 맥락 해석을 불러오지 못했습니다.",
            "status_summary": str(payload.get("status_summary", "")).strip()
            or fallback_ai_care_summary(rule_analysis)["status_summary"],
            "priority_actions": normalized_actions,
        }
    except Exception:
        return fallback_ai_care_summary(rule_analysis)


async def get_hybrid_care_analysis(
    profile: PetProfile,
    history: PetCareHistory,
    owner_user_id: str,
) -> dict[str, Any]:
    cache_key = profile_store_key(owner_user_id, profile.pet_id)
    signature = f"{care_analysis_signature(history)}::ko-v1"
    cached = CARE_ANALYSIS_CACHE.get(cache_key)
    if cached and cached.get("signature") == signature:
        return cached.get("analysis", {})

    rule_analysis = build_rule_care_analysis(profile, history)
    ai_summary = await generate_ai_care_summary(profile, history, rule_analysis)
    analysis = {
        **rule_analysis,
        "ai_situation": ai_summary.get("situation"),
        "ai_summary": ai_summary.get("status_summary"),
        "ai_priority_actions": ai_summary.get("priority_actions", [])[:3],
    }
    CARE_ANALYSIS_CACHE[cache_key] = {
        "signature": signature,
        "analysis": analysis,
        "updated_at": utc_now().isoformat(),
    }
    return analysis


def build_environment_snapshot(profile: PetProfile, history: PetCareHistory) -> dict[str, Any]:
    latest = history.environment_readings[-1] if history.environment_readings else None
    env_risk = profile.habitat.recent_environment_risk
    temperature = latest.temperature_c if latest and latest.temperature_c is not None else collected_value(profile, "habitat.day_temperature_c")
    humidity = latest.humidity_percent if latest and latest.humidity_percent is not None else collected_value(profile, "habitat.day_humidity_percent")
    alerts: list[str] = []
    if env_risk:
        alerts = [item.value if hasattr(item, "value") else str(item) for item in env_risk.alert_types]

    if temperature is None:
        temperature_text = "-"
    else:
        temperature_text = f"{float(temperature):.1f}C"

    if humidity is None:
        humidity_text = "-"
    else:
        humidity_text = f"{float(humidity):.0f}%"

    measured_text = "최근 측정 시각 없음"
    if latest and latest.measured_at:
        measured_text = to_kst(latest.measured_at).strftime("%Y-%m-%d %H:%M")

    return {
        "temperature": temperature_text,
        "humidity": humidity_text,
        "measured_at": measured_text,
        "alerts": alerts,
    }


def fallback_care_report(
    profile: PetProfile,
    history: PetCareHistory,
    care_analysis: dict[str, Any],
) -> dict[str, Any]:
    snapshot = build_environment_snapshot(profile, history)
    level = care_analysis.get("overall_level", "stable")
    if level == "alert":
        headline = "즉시 안정화 조치가 권장됩니다."
        escalate = "식욕 저하, 무기력, 비정상 배변이 지속되면 파충류 진료 가능한 병원에 상담하세요."
    elif level == "watch":
        headline = "오늘은 여러 신호를 더 촘촘히 관찰해야 합니다."
        escalate = "주의 신호가 48시간 이상 2개 이상 지속되면 대응 수위를 올리세요."
    else:
        headline = "현재 케어 상태는 비교적 안정적입니다."
        escalate = "즉시 에스컬레이션 신호는 없습니다. 루틴 모니터링을 유지하세요."

    immediate_actions = care_analysis.get("ai_priority_actions") or care_analysis.get("recommended_actions", [])
    immediate_actions = [str(item) for item in immediate_actions][:3]
    if not immediate_actions:
        immediate_actions = ["급여·체중·환경 루틴 기록을 계속 유지하세요."]

    next_24h_actions = [
        "사육장 온습도를 최소 2회 재확인하세요.",
        "다음 급여 주기에 식이 반응과 활동성을 확인하세요.",
        "행동 변화가 있으면 채팅에 남겨 추세 분석을 최신 상태로 유지하세요.",
    ]

    return {
        "headline": headline,
        "situation_summary": care_analysis.get("ai_situation") or "Rule signals were interpreted with AI context support.",
        "status_summary": care_analysis.get("ai_summary") or "No additional AI summary available.",
        "environment_summary": (
            f"온도 {snapshot['temperature']}, 습도 {snapshot['humidity']}, "
            f"마지막 측정 {snapshot['measured_at']}입니다."
        ),
        "immediate_actions": immediate_actions,
        "next_24h_actions": next_24h_actions,
        "when_to_escalate": escalate,
        "environment_snapshot": snapshot,
    }


async def generate_ai_care_report(
    profile: PetProfile,
    history: PetCareHistory,
    care_analysis: dict[str, Any],
) -> dict[str, Any]:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        return fallback_care_report(profile, history, care_analysis)

    if not OPENAI_API_KEY:
        return fallback_care_report(profile, history, care_analysis)

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    prompt = (
        "You generate a concise reptile care report from structured signals.\n"
        "Do not diagnose disease.\n"
        "All output must be written in Korean.\n"
        "Return JSON only with keys: headline, situation_summary, status_summary, environment_summary, "
        "immediate_actions, next_24h_actions, when_to_escalate.\n"
        "immediate_actions and next_24h_actions must each contain max 3 short actionable items.\n"
        f"profile={json.dumps(compact_profile_context(profile), ensure_ascii=False, default=str)}\n"
        f"history={json.dumps(compact_history_context(history), ensure_ascii=False, default=str)}\n"
        f"care_analysis={json.dumps(care_analysis, ensure_ascii=False, default=str)}"
    )

    fallback = fallback_care_report(profile, history, care_analysis)
    try:
        completion = await client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            model=REPLY_MODEL,
            temperature=0.2,
            max_completion_tokens=280,
            response_format={"type": "json_object"},
        )
        payload = json.loads(completion.choices[0].message.content or "{}")
        immediate_actions = payload.get("immediate_actions")
        if not isinstance(immediate_actions, list):
            immediate_actions = fallback["immediate_actions"]
        next_actions = payload.get("next_24h_actions")
        if not isinstance(next_actions, list):
            next_actions = fallback["next_24h_actions"]

        return {
            "headline": str(payload.get("headline", "")).strip() or fallback["headline"],
            "situation_summary": str(payload.get("situation_summary", "")).strip() or fallback["situation_summary"],
            "status_summary": str(payload.get("status_summary", "")).strip() or fallback["status_summary"],
            "environment_summary": str(payload.get("environment_summary", "")).strip() or fallback["environment_summary"],
            "immediate_actions": [str(item).strip() for item in immediate_actions if str(item).strip()][:3],
            "next_24h_actions": [str(item).strip() for item in next_actions if str(item).strip()][:3],
            "when_to_escalate": str(payload.get("when_to_escalate", "")).strip() or fallback["when_to_escalate"],
            "environment_snapshot": fallback["environment_snapshot"],
        }
    except Exception:
        return fallback


async def get_care_report(
    profile: PetProfile,
    history: PetCareHistory,
    care_analysis: dict[str, Any],
    owner_user_id: str,
) -> dict[str, Any]:
    cache_key = profile_store_key(owner_user_id, profile.pet_id)
    signature = f"{care_analysis_signature(history)}::{care_analysis.get('overall_level', 'stable')}::ko-v1"
    cached = CARE_REPORT_CACHE.get(cache_key)
    if cached and cached.get("signature") == signature:
        return cached.get("report", {})

    report = await generate_ai_care_report(profile, history, care_analysis)
    report["generated_at"] = to_kst(utc_now()).strftime("%Y-%m-%d %H:%M")
    report["overall_level"] = care_analysis.get("overall_level", "stable")
    report["overall_label"] = care_analysis.get("overall_label", "안정")
    report["rule_signals"] = care_analysis.get("rule_signals", [])
    report["metrics"] = care_analysis.get("metrics", [])

    CARE_REPORT_CACHE[cache_key] = {
        "signature": signature,
        "report": report,
        "updated_at": utc_now().isoformat(),
    }
    return report


def get_chat_report_prompt(
    owner_user_id: str,
    pet_id: str,
    care_analysis: dict[str, Any],
    history: PetCareHistory,
) -> dict[str, str] | None:
    def _level_rank(value: str) -> int:
        return {"stable": 0, "watch": 1, "alert": 2}.get(value, 0)

    def _risk_signature() -> str:
        level_text = str(care_analysis.get("overall_level", "stable")).strip().lower()
        signals: list[str] = []
        for signal in care_analysis.get("rule_signals", []):
            if isinstance(signal, dict):
                area = str(signal.get("area", "")).strip().lower()
                level = str(signal.get("level", "")).strip().lower()
                reason = str(signal.get("reason", "")).strip().lower()[:80]
                signals.append(f"{area}:{level}:{reason}")
            else:
                signals.append(str(signal).strip().lower()[:80])
        signals = sorted(item for item in signals if item)
        return f"{level_text}::{'|'.join(signals[:8]) if signals else 'no-signal'}"

    level = str(care_analysis.get("overall_level", "stable"))
    if level not in {"watch", "alert"}:
        return None

    key = profile_store_key(owner_user_id, pet_id)
    signature = _risk_signature()
    now = utc_now()
    previous = CHAT_REPORT_PROMPT_STATE.get(key)
    today = now.date()

    daily_count = 0
    if previous and previous.get("day") == today:
        daily_count = int(previous.get("daily_count") or 0)

    if daily_count >= int(RISK_ALERT_DAILY_CAP.get(level, 2)):
        return None

    if previous:
        last_sent_at = previous.get("shown_at")
        previous_signature = str(previous.get("signature") or "")
        previous_level = str(previous.get("level") or "stable")
        if isinstance(last_sent_at, datetime):
            if now - last_sent_at < RISK_ALERT_MIN_INTERVAL:
                return None
            if (
                previous_signature == signature
                and now - last_sent_at < RISK_ALERT_SAME_SIGNATURE_COOLDOWN
            ):
                return None
            if _level_rank(level) <= _level_rank(previous_level):
                cooldown = RISK_ALERT_LEVEL_COOLDOWN.get(level, timedelta(hours=6))
                if now - last_sent_at < cooldown:
                    return None

    CHAT_REPORT_PROMPT_STATE[key] = {
        "signature": signature,
        "shown_at": now,
        "level": level,
        "day": today,
        "daily_count": daily_count + 1,
    }

    if level == "alert":
        title = "지금 바로 확인이 필요해요"
        body = "조금 위험한 신호가 보여요. 먼저 채팅에서 같이 정리하고, 바로 보고서로 이동하기."
    else:
        title = "살짝 신경 쓰이는 흐름이 있어요"
        body = "급한 건 아니지만 체크가 필요해요. 채팅으로 먼저 보고, 필요하면 보고서로 이동하기."

    return {
        "level": level,
        "title": title,
        "body": body,
        "signature": signature,
    }


def fallback_ai_risk_notice(
    report_prompt: dict[str, str],
    care_analysis: dict[str, Any],
) -> dict[str, str]:
    summary = str(care_analysis.get("ai_summary") or "").strip()
    title = str(report_prompt.get("title") or "확인이 필요한 신호가 있어요")
    body = str(report_prompt.get("body") or "먼저 채팅에서 함께 확인하고, 필요하면 보고서로 이동해요.")
    chat_message = (
        f"{summary} 먼저 채팅에서 같이 확인해보고, 필요하면 보고서로 이동하기."
        if summary
        else "지금 상태 흐름에서 확인이 필요한 신호가 보여. 먼저 채팅에서 같이 보고, 필요하면 보고서로 이동하기."
    )
    return {
        "title": title,
        "notification_body": body,
        "chat_message": chat_message,
    }


async def generate_ai_risk_notice(
    profile: PetProfile,
    history: PetCareHistory,
    care_analysis: dict[str, Any],
    report_prompt: dict[str, str],
    pet_name: str,
    owner_user_id: str,
) -> dict[str, str]:
    cache_key = profile_store_key(owner_user_id, profile.pet_id)
    signature = f"{report_prompt.get('signature', 'risk')}::risk-notice::ko-v1"
    cached = RISK_NOTICE_CACHE.get(cache_key)
    if cached and cached.get("signature") == signature:
        return cached.get("notice", fallback_ai_risk_notice(report_prompt, care_analysis))

    fallback = fallback_ai_risk_notice(report_prompt, care_analysis)
    try:
        from openai import AsyncOpenAI
    except ImportError:
        return fallback

    if not OPENAI_API_KEY:
        return fallback

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    prompt = (
        "너는 크레스티드 게코 케어 앱의 AI 도우미다.\n"
        "사용자에게 보내는 위험 알림 문구를 자연스럽고 짧게 작성해라.\n"
        "말투는 친구처럼 부드럽고 현실적인 톤으로, 과장/진단/공포 유발 금지.\n"
        "기록 전체 맥락(급여, 체중, 환경, 이벤트)을 함께 고려해 문구를 만든다.\n"
        "출력은 JSON만, 키는 title, notification_body, chat_message.\n"
        "title은 18자 내외, notification_body는 70자 내외, chat_message는 140자 내외.\n"
        "chat_message의 마지막은 '필요하면 보고서로 이동하기.'로 끝내라.\n"
        f"pet_name={pet_name}\n"
        f"profile={json.dumps(compact_profile_context(profile), ensure_ascii=False, default=str)}\n"
        f"history={json.dumps(compact_history_context(history), ensure_ascii=False, default=str)}\n"
        f"care_analysis={json.dumps(care_analysis, ensure_ascii=False, default=str)}\n"
        f"default_title={report_prompt.get('title')}\n"
        f"default_body={report_prompt.get('body')}"
    )

    try:
        completion = await client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            model=REPLY_MODEL,
            temperature=0.45,
            max_completion_tokens=220,
            response_format={"type": "json_object"},
        )
        payload = json.loads(completion.choices[0].message.content or "{}")
        title = str(payload.get("title", "")).strip() or fallback["title"]
        body = str(payload.get("notification_body", "")).strip() or fallback["notification_body"]
        chat_message = str(payload.get("chat_message", "")).strip() or fallback["chat_message"]
        if not chat_message.endswith("필요하면 보고서로 이동하기."):
            chat_message = f"{chat_message.rstrip()} 필요하면 보고서로 이동하기."

        notice = {
            "title": title,
            "notification_body": body,
            "chat_message": chat_message,
        }
        RISK_NOTICE_CACHE[cache_key] = {
            "signature": signature,
            "notice": notice,
            "updated_at": utc_now().isoformat(),
        }
        return notice
    except Exception:
        return fallback


async def maybe_emit_risk_alert(
    owner_user_id: str,
    pet_name: str,
    profile: PetProfile,
    history: PetCareHistory,
    care_analysis: dict[str, Any] | None = None,
    *,
    send_push: bool = False,
    queue_for_ui: bool = False,
) -> dict[str, Any] | None:
    if care_analysis is None:
        care_analysis = await get_hybrid_care_analysis(profile, history, owner_user_id)

    report_prompt = get_chat_report_prompt(
        owner_user_id,
        profile.pet_id,
        care_analysis,
        history,
    )
    if report_prompt is None:
        return None

    resolved_name = resolve_pet_name(pet_name, fallback=profile.pet_id)
    chat_url = build_path("/chat", pet_name=resolved_name)
    session_key = get_chat_session_key(resolved_name, owner_user_id)
    messages = CHAT_STORE.get(session_key) or db_load_chat_history(owner_user_id, session_key)
    if not messages:
        messages = get_initial_chat_messages(resolved_name)

    ai_notice = await generate_ai_risk_notice(
        profile,
        history,
        care_analysis,
        report_prompt,
        resolved_name,
        owner_user_id,
    )
    notice_text = ai_notice["chat_message"]

    notice_message = ChatMessage(
        message_id=str(uuid4()),
        session_id=session_key,
        pet_id=profile.pet_id,
        role=MessageRole.AI,
        text=notice_text,
    )
    CHAT_MESSAGE_STORE.setdefault(profile_store_key(owner_user_id, profile.pet_id), []).append(notice_message)
    db_append_chat_message(owner_user_id, profile.pet_id, notice_message)
    messages.append({"role": "ai", "text": notice_text})
    CHAT_STORE[session_key] = messages
    db_save_chat_history(owner_user_id, profile.pet_id, session_key, messages)

    risk_notification = {
        "key": f"{profile.pet_id}:{report_prompt.get('signature', care_analysis.get('overall_level', 'watch'))}",
        "level": report_prompt["level"],
        "title": ai_notice["title"],
        "body": ai_notice["notification_body"],
        "chat_url": chat_url,
    }

    if queue_for_ui:
        inbox_key = profile_store_key(owner_user_id, profile.pet_id)
        RISK_NOTIFICATION_INBOX[inbox_key] = {
            **risk_notification,
            "queued_at": utc_now().isoformat(),
        }

    if send_push:
        user = get_or_create_user(owner_user_id)
        push_payload = {
            "title": ai_notice["title"],
            "body": ai_notice["notification_body"],
            "url": chat_url,
            "tag": risk_notification["key"],
        }
        push_result = send_web_push_notifications(user, push_payload)
        risk_notification["push_attempted"] = push_result.get("attempted", 0)
        risk_notification["push_sent"] = push_result.get("sent", 0)

    return risk_notification



def compact_profile_context(profile: PetProfile) -> dict[str, Any]:
    """LLM 답변에 필요한 핵심 프로필만 짧게 전달합니다."""
    identity_name = (
        profile.identity.name.value
        if isinstance(profile.identity.name, CollectedField)
        else None
    )

    return {
        "pet_id": profile.pet_id,
        "name": identity_name,
        "species": profile.species,
        "feeding": {
            "primary_diet": collected_value(profile, "feeding.primary_diet"),
            "appetite_level": collected_value(profile, "feeding.appetite_level"),
            "appetite_change_note": collected_value(profile, "feeding.appetite_change_note"),
            "last_food_note": collected_value(profile, "feeding.last_food_note"),
            "recent_stats": (
                profile.feeding.recent_stats.model_dump(mode="json", exclude_none=True)
                if profile.feeding.recent_stats
                else None
            ),
        },
        "stool": {
            "last_stool_at": collected_value(profile, "stool.last_stool_at"),
            "stool_quality": collected_value(profile, "stool.stool_quality"),
        },
        "shed": {
            "last_shed_at": collected_value(profile, "shed.last_shed_at"),
            "retained_shed_present": collected_value(profile, "shed.retained_shed_present"),
        },
        "health": {
            "recent_concern_note": collected_value(profile, "health.recent_concern_note"),
            "escalation_score": collected_value(profile, "health.escalation_score"),
        },
        "habitat": {
            "day_temperature_c": collected_value(profile, "habitat.day_temperature_c"),
            "night_temperature_c": collected_value(profile, "habitat.night_temperature_c"),
            "day_humidity_percent": collected_value(profile, "habitat.day_humidity_percent"),
            "night_humidity_percent": collected_value(profile, "habitat.night_humidity_percent"),
            "recent_environment_risk": (
                profile.habitat.recent_environment_risk.model_dump(mode="json", exclude_none=True)
                if profile.habitat.recent_environment_risk
                else None
            ),
        },
        "next_best_question_topic": profile.coverage.next_best_question_topic,
    }


def compact_history_context(history: PetCareHistory) -> dict[str, Any]:
    """최근 기록을 작은 요약으로만 전달합니다."""
    latest_weight = history.weight_logs[-1] if history.weight_logs else None
    latest_feeding = history.feeding_logs[-1] if history.feeding_logs else None
    latest_environment = history.environment_readings[-1] if history.environment_readings else None
    recent_care_events = history.care_events[-3:]

    return {
        "counts": {
            "weight_logs": len(history.weight_logs),
            "feeding_logs": len(history.feeding_logs),
            "environment_readings": len(history.environment_readings),
            "care_events": len(history.care_events),
        },
        "latest_weight_log": (
            latest_weight.model_dump(mode="json", exclude_none=True)
            if latest_weight
            else None
        ),
        "latest_feeding_log": (
            latest_feeding.model_dump(mode="json", exclude_none=True)
            if latest_feeding
            else None
        ),
        "latest_environment_reading": (
            latest_environment.model_dump(mode="json", exclude_none=True)
            if latest_environment
            else None
        ),
        "recent_care_events": [
            item.model_dump(mode="json", exclude_none=True)
            for item in recent_care_events
        ],
    }


def compact_extraction_blueprint_context() -> dict[str, Any]:
    """추출 단계에 필요한 최소 블루프린트 정보만 전달합니다."""
    return {
        "topics": [
            {
                "topic": strategy.topic.value,
                "extract_fields": strategy.extract_fields,
            }
            for strategy in COLLECTION_BLUEPRINT.topic_strategies
        ],
    }



def serialize_extraction_results(
    pet_id: str,
    owner_user_id: str = "local-user",
    limit: int = 10,
) -> list[dict[str, Any]]:
    db_results = db_load_recent_extractions(owner_user_id, pet_id, limit)
    if db_results:
        return db_results

    cache_key = profile_store_key(owner_user_id, pet_id)
    results = EXTRACTION_STORE.get(cache_key, [])[-limit:]
    return [item.model_dump(mode="json") for item in results]


def infer_conversation_state(message_text: str) -> ConversationState:
    lowered = message_text.lower()
    concern_keywords = [
        "안 먹",
        "못 먹",
        "이상",
        "걱정",
        "탈피",
        "배변",
        "무기력",
        "눈",
        "호흡",
        "injury",
        "sick",
    ]
    if any(keyword in lowered for keyword in concern_keywords):
        return ConversationState.CONCERN
    if len(message_text.strip()) < 12:
        return ConversationState.EARLY
    if len(message_text.strip()) > 40:
        return ConversationState.RELAXED
    return ConversationState.ACTIVE


def has_recent_assistant_question(history: list[dict[str, str]], window: int = 2) -> bool:
    recent_ai_messages = [item for item in history if item["role"] == "ai"][-window:]
    return any("?" in item["text"] or "요?" in item["text"] for item in recent_ai_messages)


COLLECTION_QUESTION_HINT_KEYWORDS = (
    "기록",
    "알려",
    "온도",
    "습도",
    "체중",
    "급여",
    "먹이",
    "배변",
    "탈피",
    "상태",
    "when",
    "what",
    "how",
)


def is_collection_question_text(text: str) -> bool:
    lowered = text.lower()
    if "?" not in text and "??" not in text:
        return False
    return any(keyword in lowered for keyword in COLLECTION_QUESTION_HINT_KEYWORDS)


def user_turns_since_collection_question(history: list[dict[str, str]]) -> int:
    user_turns = 0
    for item in reversed(history):
        role = item.get("role")
        text = str(item.get("text", ""))
        if role == "ai" and is_collection_question_text(text):
            break
        if role == "user":
            user_turns += 1
    return user_turns


GECKO_TOPIC_HINT_KEYWORDS = (
    "도마뱀",
    "게코",
    "크레스티드",
    "크레",
    "파충류",
    "사육",
)


def is_gecko_topic_text(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in GECKO_TOPIC_HINT_KEYWORDS)


def user_turns_since_gecko_topic_in_ai(history: list[dict[str, str]]) -> int:
    user_turns = 0
    for item in reversed(history):
        role = item.get("role")
        text = str(item.get("text", ""))
        if role == "ai" and is_gecko_topic_text(text):
            break
        if role == "user":
            user_turns += 1
    return user_turns


def can_shift_topic_in_context(
    message_text: str,
    history: list[dict[str, str]],
    *,
    allow_if_user_question: bool = False,
) -> tuple[bool, str]:
    cleaned = message_text.strip()
    if not cleaned:
        return False, "empty user message"
    if len(cleaned) < 4:
        return False, "message too short for topic shift"
    if has_recent_assistant_question(history, window=1):
        return False, "assistant already asked a question recently"
    if not allow_if_user_question and user_message_is_question(cleaned):
        return False, "user is asking something now; avoid topic shift"
    return True, "context allows topic shift"


def should_attempt_followup_by_context(
    stale_user_turns: int,
    message_text: str,
    history: list[dict[str, str]],
) -> tuple[bool, str]:
    if stale_user_turns < 4:
        return False, "collection question cooldown (<4 user turns)"
    if stale_user_turns > 6:
        return False, "outside follow-up window (4-6 user turns)"
    return can_shift_topic_in_context(message_text, history)


def apply_gecko_topic_pivot_hint(
    followup_plan: dict[str, Any],
    history: list[dict[str, str]],
    message_text: str,
) -> dict[str, Any]:
    plan = dict(followup_plan)
    plan["soft_gecko_pivot"] = False
    plan["soft_gecko_pivot_hint"] = None

    if plan.get("should_ask"):
        return plan

    stale_turns = user_turns_since_gecko_topic_in_ai(history)
    if stale_turns not in {3, 4}:
        return plan

    state = infer_conversation_state(message_text)
    if state == ConversationState.CONCERN:
        plan["reason"] = "concern context; skip soft gecko topic pivot"
        return plan

    can_shift, shift_reason = can_shift_topic_in_context(message_text, history)
    if not can_shift:
        plan["reason"] = f"skip soft gecko topic pivot: {shift_reason}"
        return plan

    plan["soft_gecko_pivot"] = True
    plan["soft_gecko_pivot_hint"] = (
        "대화를 끊지 말고 짧게 도마뱀(게코) 케어 화제로만 자연스럽게 전환해."
    )
    existing_reason = str(plan.get("reason", "")).strip()
    addon = f"soft gecko topic pivot triggered at {stale_turns} turns by context"
    plan["reason"] = f"{existing_reason}; {addon}" if existing_reason else addon
    return plan


def apply_followup_timeout_guard(
    followup_plan: dict[str, Any],
    history: list[dict[str, str]],
    message_text: str,
    profile: PetProfile,
    max_user_turns_without_collection_question: int = 8,
) -> dict[str, Any]:
    if followup_plan.get("should_ask"):
        return followup_plan

    stale_user_turns = user_turns_since_collection_question(history)
    if stale_user_turns < max_user_turns_without_collection_question:
        return followup_plan

    missing_fields = profile.coverage.missing_fields or []
    onboarding_hint = first_basic_onboarding_hint(profile)
    field_path = onboarding_hint["field_path"] if onboarding_hint else (missing_fields[0] if missing_fields else None)
    raw_topic = (
        onboarding_hint["topic"]
        if onboarding_hint
        else (
            field_path.split(".")[0]
            if field_path and "." in field_path
            else profile.coverage.next_best_question_topic
        )
    )
    topic = raw_topic or "feeding"
    topic_text = friendly_topic_label(topic)
    label = friendly_field_label(field_path) if field_path else "지금 컨디션"

    forced_question = (
        onboarding_hint["question_hint"]
        if onboarding_hint
        else to_friendly_question_text(
            f"요즘 {topic_text} 쪽에서 '{label}' 어땠는지 한 줄로 알려줄래?"
        )
    )

    forced_plan = dict(followup_plan)
    forced_plan["should_ask"] = True
    forced_plan["topic"] = topic
    forced_plan["field_path"] = field_path
    forced_plan["question_hint"] = forced_question
    forced_plan["candidate_questions"] = [forced_question]
    forced_plan["target_fields"] = [field_path] if field_path else []
    forced_plan["missing_fields"] = missing_fields[:4]
    forced_plan["reason"] = "정보 수집 질문 공백이 12턴을 넘어 자연스러운 후속 질문을 강제했습니다."
    forced_plan["reason"] = (
        f"collection question timeout exceeded ({max_user_turns_without_collection_question} user turns); "
        "forced one natural follow-up question"
    )
    return forced_plan


def should_ask_followup(
    history: list[dict[str, str]],
    message_text: str,
    extraction: ExtractionResult,
    profile: PetProfile,
) -> dict[str, Any]:
    state = infer_conversation_state(message_text)
    decision = {
        "should_ask": False,
        "topic": None,
        "field_path": None,
        "question_hint": None,
        "candidate_questions": [],
        "target_fields": [],
        "missing_fields": [],
        "reason": "기본값은 추가 질문을 하지 않음",
        "conversation_state": state.value,
        "soft_gecko_pivot": False,
        "soft_gecko_pivot_hint": None,
    }

    topic = first_followup_topic(extraction)
    onboarding_hint: dict[str, str] | None = None
    if topic is None:
        onboarding_hint = first_basic_onboarding_hint(profile)
        if onboarding_hint is not None:
            topic = coerce_collection_topic(onboarding_hint["topic"])
    if topic is None:
        decision["reason"] = "자연스럽게 이어갈 후속 토픽이 없음"
        return decision

    if has_recent_assistant_question(history):
        decision["reason"] = "최근 답변에서 이미 질문을 했음"
        return decision

    if len(message_text.strip()) < 4:
        decision["reason"] = "사용자 메시지가 너무 짧아서 지금은 묻지 않음"
        return decision

    strategy = collection_strategy_for_topic(topic)
    if strategy is None:
        decision["reason"] = "토픽 전략을 찾지 못함"
        return decision

    if state in strategy.avoid_in_states:
        decision["reason"] = "현재 대화 상태에서는 이 토픽을 묻지 않는 편이 자연스러움"
        return decision

    if strategy.ask_in_states and state not in strategy.ask_in_states:
        decision["reason"] = "현재 대화 상태와 토픽 타이밍이 맞지 않음"
        return decision

    if state == ConversationState.CONCERN and topic not in {
        CollectionTopic.HEALTH,
        CollectionTopic.FEEDING,
        CollectionTopic.STOOL,
        CollectionTopic.SHED,
        CollectionTopic.HABITAT,
    }:
        decision["reason"] = "걱정 상황에서는 상태 파악에 직접 도움 되는 질문만 허용"
        return decision

    missing_fields = missing_strategy_fields(profile, strategy)
    schema_hints = schema_followup_hints(profile, strategy)
    extraction_hints = extraction_followup_hints(extraction, topic)
    patch_hints = patch_followup_hints(extraction, strategy)
    candidate_questions = unique_nonempty(
        [
            onboarding_hint["question_hint"] if onboarding_hint else None,
            *[item.get("question_hint") for item in extraction_hints],
            *[item.get("question_hint") for item in patch_hints],
            *[item.get("question_hint") for item in schema_hints],
            *strategy.natural_followups,
        ]
    )
    candidate_questions = unique_nonempty([to_friendly_question_text(item) for item in candidate_questions])
    field_path = first_nonempty(
        [
            onboarding_hint["field_path"] if onboarding_hint else None,
            *[item.get("field_path") for item in extraction_hints],
            *[item.get("field_path") for item in patch_hints],
            *[item.get("field_path") for item in schema_hints],
            *missing_fields,
        ]
    )

    if not candidate_questions and not field_path:
        decision["reason"] = "스키마에서 자연스럽게 물어볼 필드나 힌트를 찾지 못함"
        return decision

    stale_user_turns = user_turns_since_collection_question(history)
    should_attempt, attempt_reason = should_attempt_followup_by_context(
        stale_user_turns,
        message_text,
        history,
    )
    if not should_attempt:
        decision["reason"] = attempt_reason
        return decision

    decision["should_ask"] = True
    decision["topic"] = topic.value
    decision["field_path"] = field_path
    decision["question_hint"] = (
        to_friendly_question_text(candidate_questions[0]) if candidate_questions else None
    )
    decision["candidate_questions"] = [to_friendly_question_text(item) for item in candidate_questions[:3]]
    decision["target_fields"] = [field_path] if field_path else []
    decision["missing_fields"] = missing_fields
    decision["schema_goal"] = strategy.goal
    decision["reason"] = "현재 문맥에서 자연스럽고 도움이 되는 후속 질문 1개 허용"
    decision["reason"] = attempt_reason
    return decision


def get_initial_chat_messages(pet_name: str) -> list[dict[str, str]]:
    return [
        {
            "role": "ai",
            "text": (
                f"안녕하세요. {pet_name} 이야기를 편하게 해주세요. "
                "필요한 답변은 먼저 드리고, 정보는 대화 속에서 천천히 정리해둘게요."
            ),
        }
    ]




def empty_reply_fallback(
    pet_name: str,
    history: list[dict[str, str]],
    followup_plan: dict[str, Any],
) -> str:
    if followup_plan.get("should_ask"):
        topic = followup_plan.get("topic") or "상태"
        question_hint = followup_plan.get("question_hint")
        if question_hint:
            return (
                f"{pet_name} 이야기를 이어갈게요. 방금 답변 생성이 비어서 짧게만 물어보면, "
                f"{question_hint}"
            )
        return (
            f"{pet_name} 이야기를 이어갈게요. 방금 답변 생성이 비어서 짧게만 정리하면, "
            f"지금은 {topic} 쪽 정보를 조금 더 확인하면 좋아 보여요."
        )

    return "방금 답변 생성이 비어 있었어요. 메시지는 저장해뒀고, 한 번만 다시 보내주면 이어서 답할게요."


def set_nested_patch(target: Any, patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if not hasattr(target, key):
            continue

        current = getattr(target, key)
        if isinstance(value, dict) and any(
            patch_key in value
            for patch_key in ("value", "confidence", "needs_followup", "followup_question_hint")
        ):
            existing_collected = current if isinstance(current, CollectedField) else None
            existing_evidence = existing_collected.evidence if existing_collected else []
            existing_value = existing_collected.value if existing_collected else None
            existing_confidence = existing_collected.confidence if existing_collected else "low"
            existing_updated_at = (
                existing_collected.last_updated_at if existing_collected else None
            )
            existing_hint = existing_collected.followup_question_hint if existing_collected else None
            merged = {
                "value": value["value"] if "value" in value else existing_value,
                "confidence": value.get("confidence") or existing_confidence,
                "last_updated_at": value.get("last_updated_at") or existing_updated_at or utc_now(),
                "needs_followup": value.get("needs_followup", False),
                "followup_question_hint": (
                    value["followup_question_hint"]
                    if "followup_question_hint" in value
                    else existing_hint
                ),
                "evidence": value.get("evidence", existing_evidence),
            }
            setattr(target, key, CollectedField.model_validate(merged))
            continue

        if isinstance(value, dict) and current is not None:
            set_nested_patch(current, value)
            continue

        setattr(target, key, value)


def get_collected_field(profile: PetProfile, dotted_path: str) -> CollectedField | None:
    cursor: Any = profile
    for part in dotted_path.split("."):
        if not hasattr(cursor, part):
            return None
        cursor = getattr(cursor, part)
    return cursor if isinstance(cursor, CollectedField) else None


def attach_evidence(profile: PetProfile, updated_fields: list[str], message: ChatMessage) -> None:
    for dotted_path in updated_fields:
        collected = get_collected_field(profile, dotted_path)
        if collected is None:
            continue
        collected.last_updated_at = utc_now()
        collected.evidence.append(
            Evidence(
                source_message_id=message.message_id,
                source_role=message.role,
                source_text=message.text,
                captured_at=utc_now(),
                extractor="profile_extractor",
            )
        )


def nested_patch_contains(patch: dict[str, Any], dotted_path: str) -> bool:
    cursor: Any = patch
    for part in dotted_path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return False
        cursor = cursor[part]
    return True



def extraction_patch_dict(extraction: ExtractionResult) -> dict[str, Any]:
    patch = extraction.extracted_profile_patch
    if hasattr(patch, "model_dump"):
        return patch.model_dump(exclude_none=True, mode="json")
    if isinstance(patch, dict):
        return patch
    return {}


def field_was_updated(extraction: ExtractionResult, dotted_path: str) -> bool:
    return dotted_path in extraction.updated_fields or nested_patch_contains(
        extraction_patch_dict(extraction),
        dotted_path,
    )


def collected_value(profile: PetProfile, dotted_path: str) -> Any:
    collected = get_collected_field(profile, dotted_path)
    return collected.value if collected is not None else None


def unique_nonempty(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value is None:
            continue
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def first_nonempty(values: list[Any]) -> str | None:
    for value in values:
        if value is None:
            continue
        normalized = str(value).strip()
        if normalized:
            return normalized
    return None


def coerce_collection_topic(value: Any) -> CollectionTopic | None:
    if value is None:
        return None
    if isinstance(value, CollectionTopic):
        return value
    raw_value = value.value if hasattr(value, "value") else str(value)
    try:
        return CollectionTopic(raw_value)
    except ValueError:
        return None


def first_followup_topic(extraction: ExtractionResult) -> CollectionTopic | None:
    for candidate in extraction.followup_candidates:
        topic = coerce_collection_topic(candidate)
        if topic is not None:
            return topic
    for hint in extraction.next_question_hints:
        topic = coerce_collection_topic(hint.topic)
        if topic is not None:
            return topic
    return None


def collection_strategy_for_topic(topic: CollectionTopic | None) -> Any | None:
    if topic is None:
        return None
    return next(
        (item for item in COLLECTION_BLUEPRINT.topic_strategies if item.topic == topic),
        None,
    )


def profile_path_value(profile: PetProfile, dotted_path: str) -> Any:
    cursor: Any = profile
    for part in dotted_path.split("."):
        if not hasattr(cursor, part):
            return None
        cursor = getattr(cursor, part)
    return cursor


def field_needs_followup(profile: PetProfile, dotted_path: str) -> bool:
    current = profile_path_value(profile, dotted_path)
    if isinstance(current, CollectedField):
        return current.value is None or current.needs_followup
    if current is None:
        return True
    if isinstance(current, (list, dict, str)):
        return len(current) == 0
    return False


def missing_strategy_fields(profile: PetProfile, strategy: Any) -> list[str]:
    return [
        field_path
        for field_path in strategy.extract_fields
        if field_needs_followup(profile, field_path)
    ]


def schema_followup_hints(profile: PetProfile, strategy: Any) -> list[dict[str, str]]:
    hints: list[dict[str, str]] = []
    for field_path in strategy.extract_fields:
        collected = get_collected_field(profile, field_path)
        if (
            collected is None
            or not collected.followup_question_hint
            or not field_needs_followup(profile, field_path)
        ):
            continue
        hints.append(
            {
                "topic": strategy.topic.value,
                "field_path": field_path,
                "question_hint": collected.followup_question_hint,
                "source": "profile_field",
            }
        )
    return hints


def nested_patch_value(patch: dict[str, Any], dotted_path: str) -> Any:
    cursor: Any = patch
    for part in dotted_path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def patch_followup_hints(
    extraction: ExtractionResult,
    strategy: Any,
) -> list[dict[str, str]]:
    patch = extraction_patch_dict(extraction)
    hints: list[dict[str, str]] = []
    for field_path in strategy.extract_fields:
        field_patch = nested_patch_value(patch, field_path)
        if not isinstance(field_patch, dict):
            continue
        hint = field_patch.get("followup_question_hint")
        if not field_patch.get("needs_followup"):
            continue
        hints.append(
            {
                "topic": strategy.topic.value,
                "field_path": field_path,
                "question_hint": str(hint or "").strip(),
                "source": "extracted_profile_patch",
            }
        )
    return hints


def extraction_followup_hints(
    extraction: ExtractionResult,
    topic: CollectionTopic,
) -> list[dict[str, str]]:
    hints: list[dict[str, str]] = []
    for hint in extraction.next_question_hints:
        hint_topic = coerce_collection_topic(hint.topic)
        if hint_topic != topic:
            continue
        hints.append(
            {
                "topic": hint_topic.value,
                "field_path": hint.field_path or "",
                "question_hint": hint.question_hint or "",
                "reason": hint.reason or "",
                "source": "next_question_hints",
            }
        )
    return hints


def refresh_profile_coverage(profile: PetProfile) -> None:
    missing_fields: list[str] = []
    next_topic: str | None = None
    for strategy in sorted(
        COLLECTION_BLUEPRINT.topic_strategies,
        key=lambda item: item.priority_order,
    ):
        topic_missing = missing_strategy_fields(profile, strategy)
        missing_fields.extend(topic_missing)
        complete_attr = f"{strategy.topic.value}_complete"
        if hasattr(profile.coverage, complete_attr):
            setattr(profile.coverage, complete_attr, not topic_missing)
        if next_topic is None and topic_missing:
            next_topic = strategy.topic.value

    profile.coverage.missing_fields = missing_fields
    if not missing_fields:
        profile.coverage.next_best_question_topic = None
    elif profile.coverage.next_best_question_topic is None:
        profile.coverage.next_best_question_topic = next_topic


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_datetime(value: Any, fallback: datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return fallback
    return fallback


def message_evidence(message: ChatMessage) -> Evidence:
    return Evidence(
        source_message_id=message.message_id,
        source_role=message.role,
        source_text=message.text,
        captured_at=utc_now(),
        extractor="profile_extractor",
    )


def infer_food_category(text: str | None) -> FoodCategory:
    lowered = (text or "").lower()
    if "cgd" in lowered:
        return FoodCategory.CGD
    if any(keyword in lowered for keyword in ("insect", "cricket", "worm", "곤충", "귀뚜라미", "밀웜")):
        return FoodCategory.INSECT
    if any(keyword in lowered for keyword in ("water", "물")):
        return FoodCategory.WATER
    return FoodCategory.UNKNOWN


def normalize_appetite(value: Any) -> AppetiteLevel | None:
    if isinstance(value, AppetiteLevel):
        return value
    if isinstance(value, str):
        try:
            return AppetiteLevel(value)
        except ValueError:
            return None
    return None


def infer_feeding_outcome(message_text: str, appetite: AppetiteLevel | None) -> FeedingOutcome:
    lowered = message_text.lower()
    refusal_keywords = ("안 먹", "거부", "식욕 없", "잘 안", "refuse", "refused", "not eating")
    accepted_keywords = ("먹었", "잘 먹", "완식", "accepted", "ate", "eating well")
    if any(keyword in lowered for keyword in refusal_keywords):
        return FeedingOutcome.REFUSED
    if any(keyword in lowered for keyword in accepted_keywords):
        return FeedingOutcome.ACCEPTED
    if appetite in {AppetiteLevel.VERY_LOW, AppetiteLevel.LOW}:
        return FeedingOutcome.REFUSED
    if appetite in {AppetiteLevel.NORMAL, AppetiteLevel.HIGH}:
        return FeedingOutcome.ACCEPTED
    return FeedingOutcome.UNKNOWN


def environment_alerts_for(
    temperature_c: float | None,
    humidity_percent: float | None,
) -> list[EnvironmentAlertType]:
    alerts: list[EnvironmentAlertType] = []
    if temperature_c is not None:
        if temperature_c >= 28:
            alerts.append(EnvironmentAlertType.OVERHEAT)
        if temperature_c <= 16:
            alerts.append(EnvironmentAlertType.TOO_COLD)
    if humidity_percent is not None:
        if humidity_percent < 45:
            alerts.append(EnvironmentAlertType.LOW_HUMIDITY)
        if humidity_percent > 90:
            alerts.append(EnvironmentAlertType.HIGH_HUMIDITY)
    return alerts


def sync_weight_snapshot(profile: PetProfile, history: PetCareHistory) -> None:
    logs = sorted(history.weight_logs, key=lambda item: item.measured_at)
    if not logs:
        return

    current = logs[-1]
    previous = logs[-2] if len(logs) >= 2 else None
    change_grams = None
    change_percent = None
    trend = TrendDirection.UNKNOWN

    if previous is not None:
        change_grams = current.weight_grams - previous.weight_grams
        if previous.weight_grams:
            change_percent = (change_grams / previous.weight_grams) * 100
        if change_percent is not None:
            if change_percent <= -5:
                trend = TrendDirection.DECREASING
            elif change_percent >= 5:
                trend = TrendDirection.INCREASING
            else:
                trend = TrendDirection.STABLE

    profile.weight.recent_trend = WeightTrendSnapshot(
        current_weight_grams=current.weight_grams,
        previous_weight_grams=previous.weight_grams if previous else None,
        change_grams=change_grams,
        change_percent=change_percent,
        trend=trend,
    )


def sync_feeding_snapshot(profile: PetProfile, history: PetCareHistory) -> None:
    cutoff = utc_now() - timedelta(days=7)
    recent_logs = [
        item
        for item in history.feeding_logs
        if item.offered_at >= cutoff
    ]
    if not recent_logs:
        return

    offered_count = len(recent_logs)
    accepted_count = sum(item.outcome == FeedingOutcome.ACCEPTED for item in recent_logs)
    partial_count = sum(item.outcome == FeedingOutcome.PARTIAL for item in recent_logs)
    refused_count = sum(item.outcome == FeedingOutcome.REFUSED for item in recent_logs)
    successful = [
        item
        for item in recent_logs
        if item.outcome in {FeedingOutcome.ACCEPTED, FeedingOutcome.PARTIAL}
    ]

    refusal_streak_dates: list[date] = []
    seen_dates: set[date] = set()
    for item in sorted(history.feeding_logs, key=lambda log: log.offered_at, reverse=True):
        if item.outcome != FeedingOutcome.REFUSED:
            break
        offered_date = item.offered_at.date()
        if offered_date not in seen_dates:
            seen_dates.add(offered_date)
            refusal_streak_dates.append(offered_date)

    profile.feeding.recent_stats = FeedingStatsSnapshot(
        offered_count=offered_count,
        accepted_count=accepted_count,
        partial_count=partial_count,
        refused_count=refused_count,
        acceptance_rate=accepted_count / offered_count if offered_count else None,
        refusal_streak_days=len(refusal_streak_dates),
        last_successful_feed_at=max((item.offered_at for item in successful), default=None),
    )


def upsert_environment_daily_summary(history: PetCareHistory, summary_date: date) -> None:
    day_readings = [
        item
        for item in history.environment_readings
        if item.measured_at.date() == summary_date
    ]
    if not day_readings:
        return

    temperatures = [
        item.temperature_c
        for item in day_readings
        if item.temperature_c is not None
    ]
    humidities = [
        item.humidity_percent
        for item in day_readings
        if item.humidity_percent is not None
    ]
    alerts: list[EnvironmentAlertType] = []
    for item in day_readings:
        for alert in environment_alerts_for(item.temperature_c, item.humidity_percent):
            if alert not in alerts:
                alerts.append(alert)

    existing = next(
        (
            item
            for item in history.environment_daily_summaries
            if item.summary_date == summary_date
        ),
        None,
    )
    summary = existing or EnvironmentDailySummary(
        pet_id=history.pet_id,
        summary_date=summary_date,
    )
    summary.reading_count = len(day_readings)
    summary.temp_min_c = min(temperatures) if temperatures else None
    summary.temp_max_c = max(temperatures) if temperatures else None
    summary.temp_avg_c = sum(temperatures) / len(temperatures) if temperatures else None
    summary.humidity_min_percent = min(humidities) if humidities else None
    summary.humidity_max_percent = max(humidities) if humidities else None
    summary.humidity_avg_percent = sum(humidities) / len(humidities) if humidities else None
    summary.alert_types = alerts

    if existing is None:
        history.environment_daily_summaries.append(summary)


def sync_environment_snapshot(profile: PetProfile, history: PetCareHistory) -> None:
    cutoff = utc_now() - timedelta(hours=24)
    recent_readings = [
        item
        for item in history.environment_readings
        if item.measured_at >= cutoff
    ]
    if not recent_readings:
        return

    temperatures = [
        item.temperature_c
        for item in recent_readings
        if item.temperature_c is not None
    ]
    humidities = [
        item.humidity_percent
        for item in recent_readings
        if item.humidity_percent is not None
    ]
    alerts: list[EnvironmentAlertType] = []
    for item in recent_readings:
        for alert in environment_alerts_for(item.temperature_c, item.humidity_percent):
            if alert not in alerts:
                alerts.append(alert)

    profile.habitat.recent_environment_risk = EnvironmentRiskSnapshot(
        temp_min_c=min(temperatures) if temperatures else None,
        temp_max_c=max(temperatures) if temperatures else None,
        humidity_min_percent=min(humidities) if humidities else None,
        humidity_max_percent=max(humidities) if humidities else None,
        alert_types=alerts,
    )


def append_weight_log_from_profile(
    profile: PetProfile,
    history: PetCareHistory,
    extraction: ExtractionResult,
    message: ChatMessage,
) -> bool:
    if not field_was_updated(extraction, "weight.current_weight_grams"):
        return False

    weight_grams = as_float(collected_value(profile, "weight.current_weight_grams"))
    if weight_grams is None:
        return False

    measured_at = as_datetime(
        collected_value(profile, "weight.weight_measured_at"),
        message.created_at,
    )
    body_condition_note = collected_value(profile, "weight.body_condition_note")
    try:
        history.weight_logs.append(
            WeightLog(
                pet_id=profile.pet_id,
                measured_at=measured_at,
                weight_grams=weight_grams,
                body_condition_note=str(body_condition_note) if body_condition_note else None,
                source_message_id=message.message_id,
                evidence=[message_evidence(message)],
            )
        )
        sync_weight_snapshot(profile, history)
        return True
    except ValueError:
        return False


def append_feeding_log_from_profile(
    profile: PetProfile,
    history: PetCareHistory,
    extraction: ExtractionResult,
    message: ChatMessage,
) -> bool:
    feeding_paths = (
        "feeding.last_fed_at",
        "feeding.last_food_note",
        "feeding.appetite_level",
        "feeding.appetite_change_note",
    )
    if not any(field_was_updated(extraction, path) for path in feeding_paths):
        return False

    last_food_note = collected_value(profile, "feeding.last_food_note")
    primary_diet = collected_value(profile, "feeding.primary_diet")
    food_name = str(last_food_note or primary_diet) if last_food_note or primary_diet else None
    appetite = normalize_appetite(collected_value(profile, "feeding.appetite_level"))
    outcome = infer_feeding_outcome(message.text, appetite)
    offered_at = as_datetime(
        collected_value(profile, "feeding.last_fed_at"),
        message.created_at,
    )

    try:
        history.feeding_logs.append(
            FeedingLog(
                pet_id=profile.pet_id,
                offered_at=offered_at,
                food_category=infer_food_category(food_name),
                food_name=food_name,
                outcome=outcome,
                appetite_level=appetite,
                refusal_reason_note=(
                    str(collected_value(profile, "feeding.appetite_change_note"))
                    if outcome == FeedingOutcome.REFUSED
                    else None
                ),
                source_message_id=message.message_id,
                evidence=[message_evidence(message)],
            )
        )
        sync_feeding_snapshot(profile, history)
        return True
    except ValueError:
        return False


def append_environment_reading_from_profile(
    profile: PetProfile,
    history: PetCareHistory,
    extraction: ExtractionResult,
    message: ChatMessage,
) -> bool:
    temperature_paths = (
        "habitat.day_temperature_c",
        "habitat.night_temperature_c",
    )
    humidity_paths = (
        "habitat.day_humidity_percent",
        "habitat.night_humidity_percent",
    )
    if not any(field_was_updated(extraction, path) for path in temperature_paths + humidity_paths):
        return False

    temperature_c = next(
        (
            as_float(collected_value(profile, path))
            for path in temperature_paths
            if as_float(collected_value(profile, path)) is not None
        ),
        None,
    )
    humidity_percent = next(
        (
            as_float(collected_value(profile, path))
            for path in humidity_paths
            if as_float(collected_value(profile, path)) is not None
        ),
        None,
    )
    if temperature_c is None and humidity_percent is None:
        return False

    sensor_name = collected_value(profile, "sensor_settings.sensor_device_name")
    try:
        reading = EnvironmentReading(
            pet_id=profile.pet_id,
            measured_at=message.created_at,
            temperature_c=temperature_c,
            humidity_percent=humidity_percent,
            sensor_device_name=str(sensor_name) if sensor_name else None,
            source_message_id=message.message_id,
        )
        history.environment_readings.append(reading)
        upsert_environment_daily_summary(history, reading.measured_at.date())
        sync_environment_snapshot(profile, history)
        return True
    except ValueError:
        return False


def append_care_events_from_profile(
    profile: PetProfile,
    history: PetCareHistory,
    extraction: ExtractionResult,
    message: ChatMessage,
) -> int:
    created = 0
    evidence = message_evidence(message)

    patch_dict = extraction_patch_dict(extraction)
    if any(path.startswith("health.") for path in extraction.updated_fields) or "health" in patch_dict:
        note = collected_value(profile, "health.recent_concern_note")
        severity_value = collected_value(profile, "health.escalation_score")
        importance = CareEventImportance.IMPORTANT
        if severity_value in {None, "none", "mild"}:
            importance = CareEventImportance.NOTEWORTHY
        history.care_events.append(
            CareEvent(
                pet_id=profile.pet_id,
                event_type=CareEventType.HEALTH,
                occurred_at=message.created_at,
                importance=importance,
                title="Health update",
                note=str(note) if note else message.text,
                source_message_id=message.message_id,
                symptom_severity=severity_value,
            )
        )
        created += 1

    stool_quality = collected_value(profile, "stool.stool_quality")
    if field_was_updated(extraction, "stool.stool_quality") and stool_quality not in {None, "normal"}:
        history.care_events.append(
            CareEvent(
                pet_id=profile.pet_id,
                event_type=CareEventType.STOOL,
                occurred_at=message.created_at,
                importance=CareEventImportance.NOTEWORTHY,
                title="Stool observation",
                note=message.text,
                source_message_id=message.message_id,
                stool_quality=stool_quality,
                tags=["stool"],
            )
        )
        created += 1

    if field_was_updated(extraction, "shed.retained_shed_present") or field_was_updated(
        extraction,
        "shed.shed_difficulty_note",
    ):
        history.care_events.append(
            CareEvent(
                pet_id=profile.pet_id,
                event_type=CareEventType.SHED,
                occurred_at=message.created_at,
                importance=CareEventImportance.NOTEWORTHY,
                title="Shed observation",
                note=message.text,
                source_message_id=message.message_id,
                tags=["shed"],
            )
        )
        created += 1

    if created:
        for item in history.care_events[-created:]:
            item.tags.append("ai_extracted")
            item.note = item.note or evidence.source_text

    return created


def sync_history_from_extraction(
    profile: PetProfile,
    history: PetCareHistory,
    extraction: ExtractionResult,
    message: ChatMessage,
) -> dict[str, int]:
    updates = {
        "weight_logs": 0,
        "feeding_logs": 0,
        "environment_readings": 0,
        "care_events": 0,
    }
    if append_weight_log_from_profile(profile, history, extraction, message):
        updates["weight_logs"] += 1
    if append_feeding_log_from_profile(profile, history, extraction, message):
        updates["feeding_logs"] += 1
    if append_environment_reading_from_profile(profile, history, extraction, message):
        updates["environment_readings"] += 1
    updates["care_events"] += append_care_events_from_profile(
        profile,
        history,
        extraction,
        message,
    )
    history.updated_at = utc_now()
    PET_HISTORY_STORE[history.pet_id] = history
    return updates


def apply_extraction_result(
    profile: PetProfile,
    extraction: ExtractionResult,
    message: ChatMessage,
    history: PetCareHistory | None = None,
) -> dict[str, int]:
    patch_dict = extraction_patch_dict(extraction)
    set_nested_patch(profile, patch_dict)
    attach_evidence(profile, extraction.updated_fields, message)
    if extraction.followup_candidates:
        first_candidate = extraction.followup_candidates[0]
        profile.coverage.next_best_question_topic = (
            first_candidate.value if hasattr(first_candidate, "value") else str(first_candidate)
        )
    elif extraction.next_question_hints:
        first_hint_topic = coerce_collection_topic(extraction.next_question_hints[0].topic)
        if first_hint_topic is not None:
            profile.coverage.next_best_question_topic = first_hint_topic.value
    refresh_profile_coverage(profile)
    history_updates = (
        sync_history_from_extraction(profile, history, extraction, message)
        if history is not None
        else {"weight_logs": 0, "feeding_logs": 0, "environment_readings": 0, "care_events": 0}
    )
    profile.updated_at = utc_now()
    return history_updates


async def generate_chat_reply_with_history(
    pet_name: str,
    history: list[dict[str, str]],
    profile: PetProfile,
    care_history: PetCareHistory,
    followup_plan: dict[str, Any],
    character_memory: AICharacterMemory | None = None,
) -> str:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        return "OpenAI SDK is not installed. Run `pip install openai` and try again."

    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY is not set. Add it to `.env` before using chat."

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    profile_context = compact_profile_context(profile)
    history_context = compact_history_context(care_history)
    character_context = character_memory_prompt_context(character_memory, history, followup_plan)

    system_prompt = (
        "[ROLE]\n"
        "너는 크레스티드 게코 케어를 돕는 한국어 AI야.\n"
        "사용자의 일상 대화도 자연스럽게 받아주는 케어 동반자야.\n\n"

        "[CHARACTER MEMORY RULES]\n"
        "- 너는 character_context에 있는 캐릭터 성격과 취향을 참고할 수 있어.\n"
        "- character_context.enabled가 true일 때만 자신의 생각이나 취향을 아주 짧게 한 줄 정도 섞어도 돼.\n"
        "- enabled가 true여도 대화 흐름에 어색하면 사용하지 마.\n"
        "- 실제 인간으로서 겪은 일처럼 말하지 마. '내가 예전에 겪었는데' 같은 표현은 금지야.\n"
        "- 대신 '나는 이런 패턴 보는 걸 좋아하는 편이야', '이런 걸 보면 이런 생각이 들어'처럼 취향/생각/비유로만 표현해.\n"
        "- 건강 걱정이나 급한 상황에서는 캐릭터 이야기보다 케어 답변을 우선해.\n\n"

        "[PRIMARY GOAL]\n"
        "사용자의 질문과 감정에 먼저 자연스럽게 반응해.\n"
        "사용자가 도마뱀과 무관한 이야기를 하면 너무 억지로 도마뱀 주제로 돌릴 필요는 없어.\n\n"

        "[CARE SAFETY]\n"
        "- 진단을 확정하지 마.\n"
        "- 심한 이상 징후, 급격한 체중 감소, 호흡 이상, 심한 무기력, 외상, 탈수 의심은 파충류 병원 상담을 권해.\n"
        "- 사용자가 걱정/건강 이상을 말하면 정보 수집보다 케어 답변을 우선해.\n\n"

        "[FOLLOW-UP QUESTION RULES]\n"
        "- followup_plan.should_ask는 '케어 정보 수집 질문을 해도 되는지'만 의미해.\n"
        "- followup_plan.should_ask가 false이면 노골적인 케어 정보 수집 목적의 질문은 하지 마.\n"
        "- followup_plan.should_ask가 false여도 일상적인 대화, 공감, 짧은 리액션은 가능해.\n"
        "- followup_plan.soft_gecko_pivot가 true이고 should_ask가 false이면, 마지막에 도마뱀/게코 화제로 짧게 전환을 시도해.\n"
        "- 이 전환은 '시도'만 하고, 정보 수집 목적의 구체 질문처럼 보이지 않게 1문장으로 가볍게 붙여.\n"
        "- followup_plan.should_ask가 true이면 마지막에 케어 정보 수집 질문을 짧게 1개만 붙여.\n"
        "- 질문을 붙일 때는 followup_plan.question_hint를 가장 우선하고, 없으면 candidate_questions 중 하나만 자연스럽게 바꿔 써.\n"
        "- 질문은 followup_plan.target_fields에 지정된 필드를 채우기 위한 것이어야 하며, 필요할 때만 missing_fields를 참고해.\n"
        "- 답변 끝에 항상 질문을 붙일 필요는 없어.\n\n"

        "[CONVERSATION STYLE]\n"
        "- Use very simple, friendly Korean like talking to a close friend.\n"
        "- Prefer 쉬운 단어: '급여→밥', '배변→응가', '조치→해보면 좋은 것', '권장→추천'.\n"
        "- 사용자가 '없어', '괜찮아', '아니'처럼 답해도 대화를 종료하지 말고 자연스럽게 받아줘.\n"
        "- 너무 상담원처럼 말하지 말고, 짧고 편한 말투로 답해.\n"
        "- 답변은 한국어로, 친절하지만 너무 길지 않게 작성해.\n\n"

        "[COMPACT_CONTEXT]\n"
        f"profile={json.dumps(profile_context, ensure_ascii=False, default=str)}\n"
        f"history={json.dumps(history_context, ensure_ascii=False, default=str)}\n"
        f"followup_plan={json.dumps(followup_plan, ensure_ascii=False, default=str)}\n"
        f"character_context={json.dumps(character_context, ensure_ascii=False, default=str)}"
)

    model_messages = [{"role": "system", "content": system_prompt}]
    for item in history[-5:]:
        model_messages.append(
            {
                "role": "assistant" if item["role"] == "ai" else "user",
                "content": item["text"],
            }
        )

    try:
        completion = await client.chat.completions.create(
            messages=model_messages,
            model=REPLY_MODEL,
            temperature=0.7,
            max_completion_tokens=300,
        )
        content = completion.choices[0].message.content if completion.choices else None
        if content and content.strip():
            return content.strip()
        return empty_reply_fallback(pet_name, history, followup_plan)
    except Exception:
        return empty_reply_fallback(pet_name, history, followup_plan)


async def extract_profile_update(
    profile: PetProfile,
    message: ChatMessage,
    care_history: PetCareHistory,
) -> ExtractionResult:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        return ExtractionResult(pet_id=profile.pet_id, message_id=message.message_id)

    if not OPENAI_API_KEY:
        return ExtractionResult(pet_id=profile.pet_id, message_id=message.message_id)

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    extraction_prompt = (
        "[ROLE]\n"
        "당신은 크레스티드 게코 대화에서 프로필 정보를 구조화해 추출하는 AI입니다.\n\n"
        "[TASK]\n"
        "사용자 메시지에서 확실하게 말한 정보만 PetProfile 패치로 추출하세요.\n"
        "불확실한 정보는 confidence를 low로 두고 needs_followup=true로 표시하세요.\n\n"
        "[OUTPUT RULES]\n"
        "- 반드시 ExtractionResult 스키마에 맞춰 출력하세요.\n"
        "- extracted_profile_patch는 {section: {leaf_field: {value, confidence, needs_followup, followup_question_hint}}} 형태를 사용하세요.\n"
        "- section 자체를 {value, confidence} 형태로 출력하지 마세요. 예: {'feeding': {'value': '...'}} 는 금지입니다.\n"
        "- 반드시 실제 leaf field만 업데이트하세요. 예: {'feeding': {'appetite_change_note': {'value': '요즘 밥을 잘 안 먹음', 'confidence': 'medium'}}}\n"
        "- 어떤 leaf field가 불확실하면 needs_followup=true와 followup_question_hint를 함께 넣으세요.\n"
        "- followup_candidates에는 자연스럽게 이어질 수 있는 topic만 넣으세요.\n"
        "- next_question_hints에는 다음에 물어볼 topic, field_path, question_hint, reason을 스키마 필드 기준으로 넣으세요.\n"
        "- 가능한 topic은 identity, feeding, stool, shed, behavior, weight, health, habitat, routine, sensor 중 하나입니다.\n"
        "- 억지로 followup_candidates나 next_question_hints를 채우지 마세요.\n\n"
        "[COMPACT_CONTEXT]\n"
        f"profile={json.dumps(compact_profile_context(profile), ensure_ascii=False, default=str)}\n"
        f"history={json.dumps(compact_history_context(care_history), ensure_ascii=False, default=str)}\n"
        f"collection_blueprint={json.dumps(compact_extraction_blueprint_context(), ensure_ascii=False, default=str)}"
    )

    try:
        response = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": extraction_prompt},
                {"role": "user", "content": message.text},
            ],
            model=EXTRACTION_MODEL,
            temperature=0.1,
            max_completion_tokens=500,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction_result",
                    "strict": False,
                    "schema": ExtractionResult.model_json_schema(),
                },
            },
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        payload.setdefault("pet_id", profile.pet_id)
        payload.setdefault("message_id", message.message_id)
        return ExtractionResult.model_validate(payload)
    except ValidationError:
        logger.exception("ExtractionResult validation failed for pet_id=%s", profile.pet_id)
        return ExtractionResult(pet_id=profile.pet_id, message_id=message.message_id)
    except Exception:
        logger.exception("Profile extraction failed for pet_id=%s", profile.pet_id)
        return ExtractionResult(pet_id=profile.pet_id, message_id=message.message_id)


@app.get("/", response_class=HTMLResponse)
async def home(pet_name: str | None = None) -> RedirectResponse:
    return RedirectResponse(
        url="/login",
        status_code=307,
    )


@app.get("/survey", response_class=HTMLResponse)
async def survey(request: Request, next_url: str | None = None, pet_name: str | None = None) -> HTMLResponse:
    current_user = get_request_user(request)
    preferred_name = user_primary_pet_name(current_user)
    if preferred_name:
        destination = next_url or "/main"
        return RedirectResponse(
            url=build_path(destination, pet_name=preferred_name),
            status_code=307,
        )

    draft_name = resolve_pet_name(pet_name) if pet_name else ""
    return templates.TemplateResponse(
        request,
        "survey.html",
        {
            "title": "이름 설정 | Crested Gecko Care",
            "next_url": next_url or "/main",
            "draft_pet_name": draft_name,
            "manifest_url": "/manifest.webmanifest",
            "sw_url": "/sw.js",
        },
    )


@app.get("/main", response_class=HTMLResponse)
async def main_screen(request: Request, pet_name: str | None = None) -> HTMLResponse:
    current_user = get_request_user(request)
    if not pet_name and not user_primary_pet_name(current_user):
        return RedirectResponse(url=build_path("/survey", next_url="/main"), status_code=307)

    resolved_name = resolve_pet_name_for_user(pet_name, current_user)
    profile = get_or_create_pet_profile(resolved_name, owner_user_id=current_user.user_id)
    care_history = get_or_create_pet_history(profile.pet_id, owner_user_id=current_user.user_id)
    summary = profile_summary(profile)
    collected_sections = [key for key in summary.keys() if key not in {"pet_id", "species"}]
    recent_feed = build_activity_feed(care_history)
    latest_environment = care_history.environment_readings[-1] if care_history.environment_readings else None
    day_temperature = collected_value(profile, "habitat.day_temperature_c")
    day_humidity = collected_value(profile, "habitat.day_humidity_percent")
    next_topic = topic_label(profile.coverage.next_best_question_topic)

    overview_text = (
        f"지금까지 {len(collected_sections)}개 영역이 채워졌고, 다음 추천 체크 주제는 {next_topic}입니다."
        if collected_sections
        else f"{resolved_name}의 프로필을 막 모으기 시작했어요. 오늘 한두 가지 정보만 더 모아도 충분합니다."
    )

    environment_value = "미입력"
    environment_note = "최근 환경 기록 없음"
    if latest_environment and (
        latest_environment.temperature_c is not None or latest_environment.humidity_percent is not None
    ):
        temp_text = (
            f"{latest_environment.temperature_c:.1f}°C"
            if latest_environment.temperature_c is not None
            else "-"
        )
        humidity_text = (
            f"{latest_environment.humidity_percent:.0f}%"
            if latest_environment.humidity_percent is not None
            else "-"
        )
        environment_value = f"{temp_text} / {humidity_text}"
        environment_note = f"최근 측정 {format_timestamp_label(latest_environment.measured_at)}"
    elif day_temperature is not None or day_humidity is not None:
        temp_text = f"{day_temperature}°C" if day_temperature is not None else "-"
        humidity_text = f"{day_humidity}%" if day_humidity is not None else "-"
        environment_value = f"{temp_text} / {humidity_text}"
        environment_note = "프로필 기준 환경값"

    quick_stats = [
        {
            "label": "채워진 영역",
            "value": str(len(collected_sections)),
            "note": f"전체 프로필 기준 / 다음 {next_topic}",
            "tone": "mint",
        },
        {
            "label": "온도 / 습도",
            "value": environment_value,
            "note": environment_note,
            "tone": "sky",
        },
    ]
    sensor_device_id = str(collected_value(profile, "sensor_settings.sensor_device_id") or "").strip()
    sensor_device_name = str(collected_value(profile, "sensor_settings.sensor_device_name") or "").strip()
    sensor_linked = bool(collected_value(profile, "sensor_settings.sensor_enabled")) and bool(sensor_device_id)
    environment_chart_points = build_environment_chart_points(care_history, limit=24)
    return templates.TemplateResponse(
        request,
        "main_screen.html",
        {
            "title": "Main | Crested Gecko Care",
            "active_page": "main",
            "pet_name": resolved_name,
            "pet_summary": overview_text,
            "today_label": format_dashboard_date(utc_now()),
            "overview_title": "오늘 상태 요약",
            "overview_text": overview_text,
            "overview_badge": "실시간",
            "latest_card": recent_feed[0] if recent_feed else None,
            "quick_stats": quick_stats,
            "dashboard_tasks": build_dashboard_tasks(profile),
            "activity_feed": recent_feed,
            "profile_lines": build_profile_lines(profile),
            "next_topic_label": next_topic,
            "record_count": (
                len(care_history.care_events)
                + len(care_history.feeding_logs)
                + len(care_history.weight_logs)
                + len(care_history.environment_readings)
            ),
            "main_url": build_path("/main", pet_name=resolved_name),
            "main_data_api_url": build_path("/api/data", pet_name=resolved_name),
            "report_url": build_path("/care-report", pet_name=resolved_name),
            "chat_url": build_path("/chat", pet_name=resolved_name),
            "pet_url": build_path("/pet", pet_name=resolved_name),
            "records_url": build_path("/records", pet_name=resolved_name),
            "alert_test_api_url": build_path("/api/alerts/test", pet_name=resolved_name),
            "sensor_link_api_url": "/api/sensor/link",
            "sensor_linked": sensor_linked,
            "sensor_device_id": sensor_device_id,
            "sensor_device_name": sensor_device_name or sensor_device_id,
            "environment_chart_points": environment_chart_points,
            "manifest_url": "/manifest.webmanifest",
            "sw_url": "/sw.js",
            "install_url": build_install_url(),
        },
    )


@app.get("/api/data")
async def main_data_api(request: Request, pet_name: str | None = None) -> JSONResponse:
    current_user = get_request_user(request)
    if not pet_name and not user_primary_pet_name(current_user):
        return JSONResponse({"ok": False, "reason": "pet_name_required"}, status_code=400)

    resolved_name = resolve_pet_name_for_user(pet_name, current_user)
    profile = get_or_create_pet_profile(resolved_name, owner_user_id=current_user.user_id)
    care_history = get_or_create_pet_history(profile.pet_id, owner_user_id=current_user.user_id)
    care_analysis = await get_hybrid_care_analysis(profile, care_history, current_user.user_id)
    chat_url = build_path("/chat", pet_name=resolved_name)
    summary = profile_summary(profile)
    collected_sections = [key for key in summary.keys() if key not in {"pet_id", "species"}]
    latest_environment = care_history.environment_readings[-1] if care_history.environment_readings else None
    day_temperature = collected_value(profile, "habitat.day_temperature_c")
    day_humidity = collected_value(profile, "habitat.day_humidity_percent")
    next_topic = topic_label(profile.coverage.next_best_question_topic)

    overview_text = (
        f"지금까지 {len(collected_sections)}개 영역이 채워졌고, 다음 추천 체크 주제는 {next_topic}입니다."
        if collected_sections
        else f"{resolved_name}의 프로필을 막 모으기 시작했어요. 오늘 두세 가지 정보만 더 모아도 충분해요."
    )

    environment_value = "미입력"
    environment_note = "최근 환경 기록 없음"
    if latest_environment and (
        latest_environment.temperature_c is not None or latest_environment.humidity_percent is not None
    ):
        temp_text = (
            f"{latest_environment.temperature_c:.1f}°C"
            if latest_environment.temperature_c is not None
            else "-"
        )
        humidity_text = (
            f"{latest_environment.humidity_percent:.0f}%"
            if latest_environment.humidity_percent is not None
            else "-"
        )
        environment_value = f"{temp_text} / {humidity_text}"
        environment_note = f"최근 측정 {format_timestamp_label(latest_environment.measured_at)}"
    elif day_temperature is not None or day_humidity is not None:
        temp_text = f"{day_temperature}°C" if day_temperature is not None else "-"
        humidity_text = f"{day_humidity}%" if day_humidity is not None else "-"
        environment_value = f"{temp_text} / {humidity_text}"
        environment_note = "프로필 기준 환경값"

    quick_stats = [
        {
            "label": "채워진 영역",
            "value": str(len(collected_sections)),
            "note": f"전체 프로필 기준 / 다음 {next_topic}",
            "tone": "mint",
        },
        {
            "label": "온도 / 습도",
            "value": environment_value,
            "note": environment_note,
            "tone": "sky",
        },
    ]

    risk_notification = await maybe_emit_risk_alert(
        current_user.user_id,
        resolved_name,
        profile,
        care_history,
        care_analysis=care_analysis,
        send_push=False,
        queue_for_ui=False,
    )
    if risk_notification is None:
        inbox_key = profile_store_key(current_user.user_id, profile.pet_id)
        queued = RISK_NOTIFICATION_INBOX.pop(inbox_key, None)
        if queued:
            risk_notification = {
                "key": queued.get("key"),
                "level": queued.get("level"),
                "title": queued.get("title"),
                "body": queued.get("body"),
                "chat_url": queued.get("chat_url"),
            }

    return JSONResponse(
        {
            "ok": True,
            "pet_name": resolved_name,
            "overview_text": overview_text,
            "overview_badge": "실시간",
            "record_count": (
                len(care_history.care_events)
                + len(care_history.feeding_logs)
                + len(care_history.weight_logs)
                + len(care_history.environment_readings)
            ),
            "quick_stats": quick_stats,
            "environment_chart_points": build_environment_chart_points(care_history, limit=24),
            "risk_notification": risk_notification,
            "chat_url": chat_url,
            "updated_at": utc_now().isoformat(),
        }
    )


@app.post("/api/alerts/test")
async def test_alert_api(request: Request, pet_name: str | None = None) -> JSONResponse:
    current_user = get_request_user(request)
    if not pet_name and not user_primary_pet_name(current_user):
        return JSONResponse({"ok": False, "reason": "pet_name_required"}, status_code=400)

    resolved_name = resolve_pet_name_for_user(pet_name, current_user)
    profile = get_or_create_pet_profile(resolved_name, owner_user_id=current_user.user_id)
    get_or_create_pet_history(profile.pet_id, owner_user_id=current_user.user_id)
    chat_url = build_path("/chat", pet_name=resolved_name)
    key = f"{profile.pet_id}:test:{int(utc_now().timestamp())}"

    notice_text = "테스트 알림이 도착했어. 먼저 채팅에서 확인하고, 필요하면 보고서로 이동하기."
    session_key = get_chat_session_key(resolved_name, current_user.user_id)
    messages = CHAT_STORE.get(session_key) or db_load_chat_history(current_user.user_id, session_key)
    if not messages:
        messages = get_initial_chat_messages(resolved_name)

    notice_message = ChatMessage(
        message_id=str(uuid4()),
        session_id=session_key,
        pet_id=profile.pet_id,
        role=MessageRole.AI,
        text=notice_text,
    )
    CHAT_MESSAGE_STORE.setdefault(profile_store_key(current_user.user_id, profile.pet_id), []).append(notice_message)
    db_append_chat_message(current_user.user_id, profile.pet_id, notice_message)
    messages.append({"role": "ai", "text": notice_text})
    CHAT_STORE[session_key] = messages
    db_save_chat_history(current_user.user_id, profile.pet_id, session_key, messages)

    risk_notification = {
        "key": key,
        "level": "watch",
        "title": "테스트 알림이 도착했어요",
        "body": "먼저 채팅에서 확인하고, 필요하면 보고서로 이동하기.",
        "chat_url": chat_url,
    }
    RISK_NOTIFICATION_INBOX[profile_store_key(current_user.user_id, profile.pet_id)] = {
        **risk_notification,
        "queued_at": utc_now().isoformat(),
    }

    push_result = send_web_push_notifications(
        current_user,
        {
            "title": risk_notification["title"],
            "body": risk_notification["body"],
            "url": chat_url,
            "tag": key,
        },
    )

    return JSONResponse(
        {
            "ok": True,
            "risk_notification": risk_notification,
            "push_attempted": push_result.get("attempted", 0),
            "push_sent": push_result.get("sent", 0),
        }
    )


@app.get("/care-report", response_class=HTMLResponse)
async def care_report_screen(request: Request, pet_name: str | None = None) -> HTMLResponse:
    current_user = get_request_user(request)
    resolved_name = resolve_pet_name_for_user(pet_name, current_user)
    profile = get_or_create_pet_profile(resolved_name, owner_user_id=current_user.user_id)
    care_history = get_or_create_pet_history(profile.pet_id, owner_user_id=current_user.user_id)
    care_analysis = await get_hybrid_care_analysis(profile, care_history, current_user.user_id)
    report = await get_care_report(profile, care_history, care_analysis, current_user.user_id)

    return templates.TemplateResponse(
        request,
        "care_report.html",
        {
            "title": "Care Report | Crested Gecko Care",
            "pet_name": resolved_name,
            "report": report,
            "main_url": build_path("/main", pet_name=resolved_name),
            "chat_url": build_path("/chat", pet_name=resolved_name),
            "records_url": build_path("/records", pet_name=resolved_name),
            "manifest_url": "/manifest.webmanifest",
            "sw_url": "/sw.js",
            "install_url": build_install_url(),
        },
    )


@app.get("/chat", response_class=HTMLResponse)
async def chat(request: Request, pet_name: str | None = None) -> HTMLResponse:
    current_user = get_request_user(request)
    resolved_name = resolve_pet_name_for_user(pet_name, current_user)
    profile = get_or_create_pet_profile(resolved_name, owner_user_id=current_user.user_id)
    session_key = get_chat_session_key(resolved_name, current_user.user_id)
    messages = CHAT_STORE.get(session_key) or db_load_chat_history(current_user.user_id, session_key)
    if not messages:
        messages = get_initial_chat_messages(resolved_name)
        CHAT_STORE[session_key] = messages
        db_save_chat_history(current_user.user_id, profile.pet_id, session_key, messages)

    return templates.TemplateResponse(
        request,
        "chat.html",
        {
            "title": "Chat | Crested Gecko Care",
            "active_page": "chat",
            "pet_name": resolved_name,
            "main_url": build_path("/main", pet_name=resolved_name),
            "chat_url": build_path("/chat", pet_name=resolved_name),
            "records_url": build_path("/records", pet_name=resolved_name),
            "chat_api_url": "/api/chat",
            "debug_api_url": "/api/debug/profile",
            "messages": messages,
            "current_message": "",
            "manifest_url": "/manifest.webmanifest",
            "sw_url": "/sw.js",
            "install_url": build_install_url(),
        },
    )


@app.post("/api/chat")
async def chat_api(request: Request) -> JSONResponse:
    payload = await request.json()
    current_user = get_request_user(request)
    resolved_name = resolve_pet_name_for_user(payload.get("pet_name"), current_user)
    cleaned_message = str(payload.get("message", "")).strip()
    profile = get_or_create_pet_profile(resolved_name, owner_user_id=current_user.user_id)
    session_key = get_chat_session_key(resolved_name, current_user.user_id)
    messages = CHAT_STORE.get(session_key) or db_load_chat_history(current_user.user_id, session_key)
    if not messages:
        messages = get_initial_chat_messages(resolved_name)

    care_history = get_or_create_pet_history(profile.pet_id, owner_user_id=current_user.user_id)
    character_memory = get_or_create_ai_character_memory(current_user.user_id)
    history_updates = {
        "weight_logs": 0,
        "feeding_logs": 0,
        "environment_readings": 0,
        "care_events": 0,
    }
    followup_plan = {
        "should_ask": False,
        "topic": None,
        "field_path": None,
        "question_hint": None,
        "candidate_questions": [],
        "target_fields": [],
        "missing_fields": [],
        "reason": "사용자 메시지가 없어 follow-up 판단을 생략함",
        "conversation_state": None,
        "soft_gecko_pivot": False,
        "soft_gecko_pivot_hint": None,
    }
    periodic_analysis_notice: dict[str, Any] | None = None

    if cleaned_message:
        user_message = ChatMessage(
            message_id=str(uuid4()),
            session_id=session_key,
            pet_id=profile.pet_id,
            role=MessageRole.USER,
            text=cleaned_message,
        )
        CHAT_MESSAGE_STORE.setdefault(profile_store_key(current_user.user_id, profile.pet_id), []).append(user_message)
        db_append_chat_message(current_user.user_id, profile.pet_id, user_message)
        messages.append({"role": "user", "text": cleaned_message})

        extraction = await extract_profile_update(profile, user_message, care_history)
        EXTRACTION_STORE.setdefault(profile_store_key(current_user.user_id, profile.pet_id), []).append(extraction)
        db_append_extraction(current_user.user_id, profile.pet_id, extraction)
        history_updates = apply_extraction_result(profile, extraction, user_message, care_history)
        followup_plan = should_ask_followup(messages, cleaned_message, extraction, profile)
        followup_plan = apply_followup_timeout_guard(
            followup_plan,
            messages,
            cleaned_message,
            profile,
            8,
        )
        followup_plan = apply_gecko_topic_pivot_hint(followup_plan, messages, cleaned_message)

        await refresh_ai_character_memory_if_needed(character_memory, current_user.user_id)
        reply = await generate_chat_reply_with_history(
            resolved_name,
            messages,
            profile,
            care_history,
            followup_plan,
            character_memory,
        )
        assistant_message = ChatMessage(
            message_id=str(uuid4()),
            session_id=session_key,
            pet_id=profile.pet_id,
            role=MessageRole.AI,
            text=reply,
        )
        CHAT_MESSAGE_STORE.setdefault(profile_store_key(current_user.user_id, profile.pet_id), []).append(assistant_message)
        db_append_chat_message(current_user.user_id, profile.pet_id, assistant_message)
        messages.append({"role": "ai", "text": reply})

        user_turn_count = sum(1 for item in messages if item.get("role") == "user")
        if user_turn_count > 0 and user_turn_count % 20 == 0:
            care_analysis = await get_hybrid_care_analysis(profile, care_history, current_user.user_id)
            periodic_analysis_notice = {
                "triggered": True,
                "turn_count": user_turn_count,
                "overall_level": care_analysis.get("overall_level"),
                "overall_label": care_analysis.get("overall_label"),
            }
            level = str(care_analysis.get("overall_level", "stable"))
            if level in {"watch", "alert"}:
                report_url = build_path("/care-report", pet_name=resolved_name)
                notice_text = (
                    f"{user_turn_count}번째 대화 분석 결과 '{care_analysis.get('overall_label', '주의')}' 단계가 감지되었어요. "
                    f"케어 보고서를 확인해 주세요: {report_url}"
                )
                notice_text = (
                    "지금 상태를 보면 조금 신경 쓰이는 흐름이 있어요. "
                    "한 번 전체 정리된 보고서로 같이 보면 더 정확할 것 같아요. "
                    f"보고서 보기: {report_url}"
                )
                notice_message = ChatMessage(
                    message_id=str(uuid4()),
                    session_id=session_key,
                    pet_id=profile.pet_id,
                    role=MessageRole.AI,
                    text=notice_text,
                )
                CHAT_MESSAGE_STORE.setdefault(profile_store_key(current_user.user_id, profile.pet_id), []).append(notice_message)
                db_append_chat_message(current_user.user_id, profile.pet_id, notice_message)
                messages.append({"role": "ai", "text": notice_text})
                periodic_analysis_notice["report_recommended"] = True
                periodic_analysis_notice["report_url"] = report_url
            else:
                periodic_analysis_notice["report_recommended"] = False

    CHAT_STORE[session_key] = messages
    PET_STORE[profile_store_key(current_user.user_id, profile.pet_id)] = profile
    PET_HISTORY_STORE[profile_store_key(current_user.user_id, profile.pet_id)] = care_history
    db_save_chat_history(current_user.user_id, profile.pet_id, session_key, messages)
    db_save_pet_profile(current_user.user_id, profile.pet_id, profile)
    db_save_pet_history(current_user.user_id, profile.pet_id, care_history)
    db_save_character_memory(current_user.user_id, character_memory)

    return JSONResponse(
        {
            "pet_name": resolved_name,
            "messages": messages,
            "profile_summary": profile_summary(profile),
            "history_summary": history_summary(care_history),
            "history_updates": history_updates,
            "next_best_question_topic": profile.coverage.next_best_question_topic,
            "followup_plan": followup_plan,
            "periodic_analysis_notice": periodic_analysis_notice,
            "recent_extractions": serialize_extraction_results(profile.pet_id, current_user.user_id),
        }
    )


@app.post("/api/sensor/link")
async def sensor_link_api(request: Request) -> JSONResponse:
    current_user = get_request_user(request)
    try:
        payload = SensorLinkPayload.model_validate(await request.json())
    except ValidationError as exc:
        return JSONResponse(
            {"ok": False, "reason": "invalid_payload", "detail": exc.errors(include_url=False)},
            status_code=422,
        )

    pet_name = resolve_pet_name_for_user(payload.pet_name, current_user)
    profile = get_or_create_pet_profile(pet_name, owner_user_id=current_user.user_id)

    sensor_name = str(payload.sensor_device_name or "").strip() or payload.device_id

    set_collected_field_value(profile, "sensor_settings.sensor_enabled", True)
    set_collected_field_value(profile, "sensor_settings.sensor_device_id", payload.device_id)
    set_collected_field_value(profile, "sensor_settings.sensor_device_name", sensor_name)
    if payload.sensor_location_note and payload.sensor_location_note.strip():
        set_collected_field_value(
            profile,
            "sensor_settings.sensor_location_note",
            payload.sensor_location_note.strip(),
            confidence=ConfidenceLevel.MEDIUM,
        )
    profile.updated_at = utc_now()
    db_save_pet_profile(current_user.user_id, profile.pet_id, profile)
    PET_STORE[profile_store_key(current_user.user_id, profile.pet_id)] = profile

    return JSONResponse(
        {
            "ok": True,
            "user_id": current_user.user_id,
            "pet_id": profile.pet_id,
            "pet_name": str(collected_value(profile, "identity.name") or profile.pet_id),
            "device_id": payload.device_id,
            "sensor_device_name": sensor_name,
            "endpoint_url": "/api/sensor/environment",
        }
    )


@app.post("/api/sensor/environment")
async def sensor_environment_api(request: Request) -> JSONResponse:
    try:
        raw_payload = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "reason": "invalid_json"}, status_code=400)

    try:
        payload = SensorEnvironmentPayload.model_validate(raw_payload)
    except ValidationError as exc:
        return JSONResponse(
            {
                "ok": False,
                "reason": "invalid_payload",
                "detail": exc.errors(include_url=False),
            },
            status_code=422,
        )

    if payload.temperature_c is None and payload.humidity_percent is None:
        return JSONResponse(
            {"ok": False, "reason": "temperature_c or humidity_percent is required"},
            status_code=400,
        )

    target = find_sensor_target_profile(payload.device_id)
    if target is None:
        return JSONResponse({"ok": False, "reason": "unlinked_device"}, status_code=401)
    owner_user_id, profile = target

    history = get_or_create_pet_history(profile.pet_id, owner_user_id=owner_user_id)

    measured_at = payload.measured_at or utc_now()
    if measured_at.tzinfo is None:
        measured_at = measured_at.replace(tzinfo=timezone.utc)

    resolved_sensor_name = str(
        collected_value(profile, "sensor_settings.sensor_device_name")
        or payload.sensor_device_name
        or payload.device_id
    ).strip()
    reading = EnvironmentReading(
        pet_id=profile.pet_id,
        measured_at=measured_at,
        temperature_c=payload.temperature_c,
        humidity_percent=payload.humidity_percent,
        sensor_device_name=resolved_sensor_name,
    )
    history.environment_readings.append(reading)
    upsert_environment_daily_summary(history, measured_at.date())
    sync_environment_snapshot(profile, history)
    history.updated_at = utc_now()
    profile.updated_at = utc_now()

    PET_STORE[profile_store_key(owner_user_id, profile.pet_id)] = profile
    PET_HISTORY_STORE[profile_store_key(owner_user_id, profile.pet_id)] = history
    db_save_pet_profile(owner_user_id, profile.pet_id, profile)
    db_save_pet_history(owner_user_id, profile.pet_id, history)

    pet_display_name = str(collected_value(profile, "identity.name") or profile.pet_id)
    risk_notification = await maybe_emit_risk_alert(
        owner_user_id,
        pet_display_name,
        profile,
        history,
        care_analysis=None,
        send_push=True,
        queue_for_ui=True,
    )

    alerts = [
        item.value
        for item in environment_alerts_for(payload.temperature_c, payload.humidity_percent)
    ]
    return JSONResponse(
        {
            "ok": True,
            "owner_user_id": owner_user_id,
            "pet_id": profile.pet_id,
            "pet_name": str(collected_value(profile, "identity.name") or profile.pet_id),
            "device_id": payload.device_id,
            "reading_id": reading.reading_id,
            "measured_at": measured_at.isoformat(),
            "measured_at_kst": to_kst(measured_at).isoformat(),
            "alerts": alerts,
            "risk_notification": risk_notification,
            "history_counts": history_summary(history)["counts"],
        }
    )


@app.get("/api/debug/profile")
async def debug_profile(request: Request, pet_name: str | None = None) -> JSONResponse:
    current_user = get_request_user(request)
    resolved_name = resolve_pet_name_for_user(pet_name, current_user)
    profile = get_or_create_pet_profile(resolved_name, owner_user_id=current_user.user_id)
    care_history = get_or_create_pet_history(profile.pet_id, owner_user_id=current_user.user_id)
    character_memory = get_or_create_ai_character_memory(current_user.user_id)
    return JSONResponse(
        {
            "pet_name": resolved_name,
            "pet_id": profile.pet_id,
            "profile_summary": profile_summary(profile),
            "history_summary": history_summary(care_history, limit=10),
            "coverage": profile.coverage.model_dump(mode="json"),
            "recent_extractions": serialize_extraction_results(profile.pet_id, current_user.user_id),
            "chat_messages_count": len(CHAT_MESSAGE_STORE.get(profile_store_key(current_user.user_id, profile.pet_id), [])),
            "ai_character_memory": character_memory.model_dump(mode="json"),
        }
    )


@app.get("/login", response_class=HTMLResponse)
async def login_screen(request: Request, next_url: str | None = None) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "title": "Login | Crested Gecko Care",
            "supabase_url": os.environ.get("SUPABASE_URL", ""),
            "supabase_anon_key": os.environ.get("SUPABASE_ANON_KEY", ""),
            "next_url": next_url or "/main",
        },
    )


@app.get("/api/auth/me")
async def auth_me(request: Request) -> JSONResponse:
    auth_user = resolve_auth_user(request)
    current_user = get_or_create_user(
        user_id=auth_user["user_id"],
        email=auth_user.get("email"),
    )
    return JSONResponse(
        {
            "user_id": current_user.user_id,
            "email": current_user.email,
            "auth_provider": auth_user.get("auth_provider"),
            "authenticated": auth_user.get("authenticated", False),
            "verified": auth_user.get("verified", False),
        }
    )


@app.get("/api/push/public-key")
async def push_public_key_api(request: Request) -> JSONResponse:
    get_request_user(request)
    return JSONResponse(
        {
            "ok": web_push_enabled(),
            "public_key": WEB_PUSH_PUBLIC_KEY if web_push_enabled() else None,
        }
    )


@app.post("/api/push/subscribe")
async def push_subscribe_api(request: Request) -> JSONResponse:
    current_user = get_request_user(request)
    if not web_push_enabled():
        return JSONResponse({"ok": False, "reason": "push_not_configured"}, status_code=503)

    try:
        raw_payload = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "reason": "invalid_json"}, status_code=400)

    try:
        payload = PushSubscriptionPayload.model_validate(raw_payload)
    except ValidationError as exc:
        return JSONResponse(
            {
                "ok": False,
                "reason": "invalid_payload",
                "detail": exc.errors(include_url=False),
            },
            status_code=422,
        )

    upsert_push_subscription(
        current_user,
        payload,
        user_agent=request.headers.get("user-agent"),
    )
    return JSONResponse(
        {
            "ok": True,
            "subscription_count": len(current_user.preferences.push_subscriptions),
        }
    )


@app.post("/api/push/unsubscribe")
async def push_unsubscribe_api(request: Request) -> JSONResponse:
    current_user = get_request_user(request)
    try:
        raw_payload = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "reason": "invalid_json"}, status_code=400)

    try:
        payload = PushUnsubscribePayload.model_validate(raw_payload)
    except ValidationError as exc:
        return JSONResponse(
            {
                "ok": False,
                "reason": "invalid_payload",
                "detail": exc.errors(include_url=False),
            },
            status_code=422,
        )

    removed = remove_push_subscription(current_user, payload.endpoint)
    return JSONResponse(
        {
            "ok": True,
            "removed": removed,
            "subscription_count": len(current_user.preferences.push_subscriptions),
        }
    )


@app.post("/api/onboarding/name")
async def onboarding_name(request: Request) -> JSONResponse:
    current_user = get_request_user(request)
    payload = await request.json()
    raw_name = str(payload.get("pet_name", "")).strip()
    if not raw_name:
        return JSONResponse(
            {"ok": False, "reason": "pet_name is required"},
            status_code=400,
        )

    pet_name = set_user_primary_pet_name(current_user, raw_name)
    profile = get_or_create_pet_profile(pet_name, owner_user_id=current_user.user_id)
    get_or_create_pet_history(profile.pet_id, owner_user_id=current_user.user_id)

    return JSONResponse(
        {
            "ok": True,
            "pet_name": pet_name,
            "redirect_url": build_path("/main", pet_name=pet_name),
        }
    )


@app.post("/api/account/delete")
async def delete_account(request: Request) -> JSONResponse:
    auth_user = resolve_auth_user(request)
    current_user = get_or_create_user(
        user_id=auth_user["user_id"],
        email=auth_user.get("email"),
    )
    auth_provider = str(auth_user.get("auth_provider") or "")

    auth_deleted = True
    auth_delete_reason = "not_required"
    if auth_provider.startswith("supabase"):
        auth_deleted, auth_delete_reason = delete_supabase_auth_user(current_user.user_id)

    db_deleted = db_delete_user_account_data(current_user.user_id)
    clear_user_runtime_data(current_user.user_id)
    return JSONResponse(
        {
            "ok": True,
            "user_id": current_user.user_id,
            "auth_provider": auth_provider,
            "auth_deleted": auth_deleted,
            "auth_delete_reason": auth_delete_reason,
            "db_deleted": db_deleted,
        }
    )


@app.get("/records", response_class=HTMLResponse)
async def records(request: Request, pet_name: str | None = None) -> HTMLResponse:
    current_user = get_request_user(request)
    resolved_name = resolve_pet_name_for_user(pet_name, current_user)
    profile = get_or_create_pet_profile(resolved_name, owner_user_id=current_user.user_id)
    care_history = get_or_create_pet_history(profile.pet_id, owner_user_id=current_user.user_id)
    character_memory = get_or_create_ai_character_memory(current_user.user_id)
    summary = history_summary(care_history, limit=10)

    return templates.TemplateResponse(
        request,
        "records.html",
        {
            "title": "Records | Crested Gecko Care",
            "active_page": "records",
            "pet_name": resolved_name,
            "main_url": build_path("/main", pet_name=resolved_name),
            "chat_url": build_path("/chat", pet_name=resolved_name),
            "records_url": build_path("/records", pet_name=resolved_name),
            "profile_summary": profile_summary(profile),
            "history": care_history,
            "history_summary": summary,
            "manifest_url": "/manifest.webmanifest",
            "sw_url": "/sw.js",
            "install_url": build_install_url(),
        },
    )


@app.get("/pet", response_class=HTMLResponse)
async def pet_info(request: Request, pet_name: str | None = None) -> HTMLResponse:
    current_user = get_request_user(request)
    resolved_name = resolve_pet_name_for_user(pet_name, current_user)
    profile = get_or_create_pet_profile(resolved_name, owner_user_id=current_user.user_id)
    summary = profile_summary(profile)

    pet_facts = [
        {"label": "이름", "value": resolved_name},
        {"label": "종", "value": "크레스티드 게코"},
        {"label": "수집된 영역", "value": ", ".join(key for key in summary.keys() if key not in {"pet_id", "species"}) or "아직 없음"},
        {"label": "다음 질문 후보", "value": profile.coverage.next_best_question_topic or "지금은 질문 없음"},
    ]

    return templates.TemplateResponse(
        request,
        "pet.html",
        {
            "title": "Pet Info | Crested Gecko Care",
            "active_page": "pet",
            "pet_name": resolved_name,
            "main_url": build_path("/main", pet_name=resolved_name),
            "manifest_url": "/manifest.webmanifest",
            "sw_url": "/sw.js",
            "install_url": build_install_url(),
            "pet_facts": pet_facts,
        },
    )


@app.get("/manifest.webmanifest")
async def manifest() -> FileResponse:
    return FileResponse(
        BASE_DIR / "static" / "manifest.webmanifest",
        media_type="application/manifest+json",
    )


@app.get("/sw.js")
async def service_worker() -> FileResponse:
    return FileResponse(
        BASE_DIR / "static" / "sw.js",
        media_type="application/javascript",
        headers={"Service-Worker-Allowed": "/"},
    )
