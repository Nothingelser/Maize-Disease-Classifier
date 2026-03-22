"""
Supabase client helpers for authentication and database operations.
"""
import logging
import os
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import httpx
from supabase import Client, ClientOptions, create_client
from app.domain.records import PredictionRecord, UserRecord

logger = logging.getLogger(__name__)

def utc_now():
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)

def get_app_base_url() -> str:
    """Return the base URL used in Supabase auth email redirects."""
    return os.environ.get("APP_BASE_URL", "http://127.0.0.1:5000").rstrip("/")

class SupabaseClient:
    """Singleton Supabase client for authentication and data operations."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.url = os.environ.get("SUPABASE_URL")
        self.key = os.environ.get("SUPABASE_KEY")
        self.service_key = os.environ.get("SUPABASE_SERVICE_KEY")
        self.client: Optional[Client] = None
        self.admin_client: Optional[Client] = None

        if not self.url or not self.key:
            logger.warning("SUPABASE_URL and SUPABASE_KEY not set in environment")
            return

        self.client = self._build_client(self.key)
        if self.service_key:
            self.admin_client = self._build_client(self.service_key)

        logger.info("Supabase client initialized")

    def _build_client(self, key: str) -> Client:
        http_client = httpx.Client(http2=True, follow_redirects=True)
        options = ClientOptions(httpx_client=http_client)
        return create_client(self.url, key, options=options)

    def get_client(self) -> Optional[Client]:
        return self.client

    def is_connected(self) -> bool:
        return self.client is not None

    def has_admin_access(self) -> bool:
        return self.admin_client is not None

    def _table_client(self, use_admin: bool = False) -> Optional[Client]:
        if use_admin and self.admin_client:
            return self.admin_client
        return self.client

    def _server_table_client(self) -> Optional[Client]:
        """Use the service role for backend table access when available."""
        return self.admin_client or self.client

    def _build_user_record(self, profile: Optional[Dict]) -> Optional[UserRecord]:
        return UserRecord.from_dict(profile)

    def _build_prediction_record(self, row: Optional[Dict]) -> Optional[PredictionRecord]:
        return PredictionRecord.from_dict(row)

    def find_user_profile(self, identifier: str) -> Dict:
        table_client = self._server_table_client()
        if not table_client:
            return {"success": False, "error": "Supabase client not initialized"}

        try:
            column = "email" if "@" in identifier else "username"
            response = table_client.table("users").select("*").eq(column, identifier).limit(1).execute()
            if response.data:
                return {"success": True, "profile": response.data[0]}
            return {"success": False, "error": "User not found"}
        except Exception as exc:
            logger.error("Failed to find user profile: %s", exc)
            return {"success": False, "error": str(exc)}

    def get_user_by_username(self, username: str) -> Dict:
        return self.find_user_profile(username)

    def get_user_by_email(self, email: str) -> Dict:
        return self.find_user_profile(email)

    def get_user_profile(self, user_id: str) -> Dict:
        table_client = self._server_table_client()
        if not table_client:
            return {"success": False, "error": "Supabase client not initialized"}

        try:
            response = table_client.table("users").select("*").eq("id", user_id).limit(1).execute()
            if response.data:
                return {"success": True, "profile": response.data[0]}
            return {"success": False, "error": "User not found"}
        except Exception as exc:
            logger.error("Failed to get user profile: %s", exc)
            return {"success": False, "error": str(exc)}

    def get_user_record(self, user_id: str) -> Dict:
        profile_result = self.get_user_profile(user_id)
        if not profile_result.get("success"):
            return profile_result
        return {"success": True, "user": self._build_user_record(profile_result["profile"])}

    def count_users(self) -> Dict:
        table_client = self._server_table_client()
        if not table_client:
            return {"success": False, "error": "Supabase client not initialized"}

        try:
            response = table_client.table("users").select("id", count="exact").limit(1).execute()
            return {"success": True, "count": response.count or 0}
        except Exception as exc:
            logger.error("Failed to count users: %s", exc)
            return {"success": False, "error": str(exc)}

    def sync_user_profile(
        self,
        user_id: str,
        email: str,
        username: str,
        full_name: str = None,
        is_admin: bool = False,
        avatar_url: str = None,
        last_login: str = None,
    ) -> Dict:
        table_client = self._server_table_client()
        if not table_client:
            return {"success": False, "error": "Supabase client not initialized"}

        profile_client = self.admin_client or self.client
        payload = {
            "id": user_id,
            "email": email,
            "username": username,
            "full_name": full_name,
            "is_admin": is_admin,
            "avatar_url": avatar_url,
            "updated_at": utc_now().isoformat(),
        }
        if last_login:
            payload["last_login"] = last_login

        try:
            existing = profile_client.table("users").select("id").eq("id", user_id).limit(1).execute()
            if existing.data:
                response = profile_client.table("users").update(payload).eq("id", user_id).execute()
            else:
                payload["created_at"] = utc_now().isoformat()
                response = profile_client.table("users").insert(payload).execute()
                profile_client.table("user_settings").insert({"user_id": user_id}).execute()
            return {"success": True, "profile": response.data[0] if response.data else payload}
        except Exception as exc:
            logger.error("Failed to sync user profile: %s", exc)
            return {"success": False, "error": str(exc)}

    def get_predictions(
        self,
        user_id: str,
        is_admin: bool = False,
        page: int = 1,
        per_page: int = 20,
        days: Optional[str] = None,
        disease: Optional[str] = None,
        confidence: Optional[str] = None,
    ) -> Dict:
        table_client = self._server_table_client()
        if not table_client:
            return {"success": False, "error": "Supabase client not initialized"}

        start = max(page - 1, 0) * per_page
        end = start + per_page - 1

        try:
            query = table_client.table("predictions").select("*", count="exact")
            if not is_admin:
                query = query.eq("user_id", user_id)

            if days and str(days).lower() != "all":
                try:
                    cutoff = utc_now() - timedelta(days=int(days))
                    query = query.gte("created_at", cutoff.isoformat())
                except ValueError:
                    pass

            if disease and disease != "all":
                query = query.eq("prediction", disease)

            if confidence == "high":
                query = query.gte("confidence", 0.8)
            elif confidence == "medium":
                query = query.gte("confidence", 0.5).lt("confidence", 0.8)
            elif confidence == "low":
                query = query.lt("confidence", 0.5)

            response = query.order("created_at", desc=True).range(start, end).execute()
            predictions = [self._build_prediction_record(row) for row in (response.data or [])]
            total = response.count or 0
            return {
                "success": True,
                "predictions": predictions,
                "total": total,
                "pages": (total + per_page - 1) // per_page if per_page else 1,
                "current_page": page,
            }
        except Exception as exc:
            logger.error("Failed to fetch predictions: %s", exc)
            return {"success": False, "error": str(exc)}

    def count_user_predictions(self, user_id: str) -> Dict:
        return self.get_predictions(user_id=user_id, page=1, per_page=1)

    def get_prediction(self, prediction_id: int, user_id: str, is_admin: bool = False) -> Dict:
        table_client = self._server_table_client()
        if not table_client:
            return {"success": False, "error": "Supabase client not initialized"}

        try:
            query = table_client.table("predictions").select("*").eq("id", prediction_id)
            if not is_admin:
                query = query.eq("user_id", user_id)
            response = query.limit(1).execute()
            if not response.data:
                return {"success": False, "error": "Prediction not found"}
            return {"success": True, "prediction": self._build_prediction_record(response.data[0])}
        except Exception as exc:
            logger.error("Failed to fetch prediction: %s", exc)
            return {"success": False, "error": str(exc)}

    def create_prediction(
        self,
        user_id: str,
        image_name: str,
        prediction: str,
        confidence: float,
        probabilities: List[Dict],
        processing_time: Optional[float] = None,
        image_url: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Dict:
        table_client = self._server_table_client()
        if not table_client:
            return {"success": False, "error": "Supabase client not initialized"}

        payload = {
            "user_id": user_id,
            "image_name": image_name,
            "image_url": image_url,
            "prediction": prediction,
            "confidence": confidence,
            "probabilities_json": json.dumps(probabilities),
            "processing_time": processing_time,
            "ip_address": ip_address,
            "user_agent": user_agent,
        }

        try:
            response = table_client.table("predictions").insert(payload).execute()
            if not response.data:
                return {"success": False, "error": "Prediction was not stored"}
            return {"success": True, "prediction": self._build_prediction_record(response.data[0])}
        except Exception as exc:
            logger.error("Failed to create prediction: %s", exc)
            return {"success": False, "error": str(exc)}

    def get_predictions_by_ids(self, user_id: str, prediction_ids: List[int], is_admin: bool = False) -> Dict:
        table_client = self._server_table_client()
        if not table_client:
            return {"success": False, "error": "Supabase client not initialized"}
        if not prediction_ids:
            return {"success": True, "predictions": []}

        try:
            query = table_client.table("predictions").select("*").in_("id", prediction_ids)
            if not is_admin:
                query = query.eq("user_id", user_id)
            response = query.order("created_at", desc=True).execute()
            predictions = [self._build_prediction_record(row) for row in (response.data or [])]
            return {"success": True, "predictions": predictions}
        except Exception as exc:
            logger.error("Failed to fetch predictions for export: %s", exc)
            return {"success": False, "error": str(exc)}

    def list_predictions_since(self, user_id: Optional[str], start_iso: str, end_iso: Optional[str] = None) -> Dict:
        table_client = self._server_table_client()
        if not table_client:
            return {"success": False, "error": "Supabase client not initialized"}

        try:
            query = table_client.table("predictions").select("*").gte("created_at", start_iso)
            if end_iso:
                query = query.lt("created_at", end_iso)
            if user_id:
                query = query.eq("user_id", user_id)
            response = query.order("created_at", desc=False).execute()
            predictions = [self._build_prediction_record(row) for row in (response.data or [])]
            return {"success": True, "predictions": predictions}
        except Exception as exc:
            logger.error("Failed to list predictions: %s", exc)
            return {"success": False, "error": str(exc)}

    def get_prediction_distribution(self) -> Dict:
        table_client = self._server_table_client()
        if not table_client:
            return {"success": False, "error": "Supabase client not initialized"}

        try:
            response = table_client.table("predictions").select("prediction").execute()
            counts: Dict[str, int] = {}
            for row in response.data or []:
                label = row.get("prediction")
                if label:
                    counts[label] = counts.get(label, 0) + 1
            return {
                "success": True,
                "distribution": [
                    {"disease": disease, "count": count}
                    for disease, count in sorted(counts.items())
                ],
                "total": sum(counts.values()),
            }
        except Exception as exc:
            logger.error("Failed to build prediction distribution: %s", exc)
            return {"success": False, "error": str(exc)}

    def count_predictions(self) -> Dict:
        table_client = self._server_table_client()
        if not table_client:
            return {"success": False, "error": "Supabase client not initialized"}

        try:
            response = table_client.table("predictions").select("id", count="exact").limit(1).execute()
            return {"success": True, "count": response.count or 0}
        except Exception as exc:
            logger.error("Failed to count predictions: %s", exc)
            return {"success": False, "error": str(exc)}

    def count_system_logs_since(self, level: str, start_iso: str) -> Dict:
        table_client = self._server_table_client()
        if not table_client:
            return {"success": False, "error": "Supabase client not initialized"}

        try:
            response = (
                table_client.table("system_logs")
                .select("id", count="exact")
                .eq("level", level)
                .gte("created_at", start_iso)
                .limit(1)
                .execute()
            )
            return {"success": True, "count": response.count or 0}
        except Exception as exc:
            logger.warning("Failed to count system logs: %s", exc)
            return {"success": False, "error": str(exc), "count": 0}

    def ping(self) -> Dict:
        table_client = self._server_table_client()
        if not table_client:
            return {"success": False, "error": "Supabase client not initialized"}

        try:
            table_client.table("users").select("id").limit(1).execute()
            return {"success": True}
        except Exception as exc:
            logger.error("Supabase health check failed: %s", exc)
            return {"success": False, "error": str(exc)}

    def register_auth_user(
        self,
        email: str,
        password: str,
        username: str,
        full_name: str = None,
        is_admin: bool = False,
    ) -> Dict:
        if not self.client:
            return {"success": False, "error": "Supabase client not initialized"}

        try:
            app_base_url = get_app_base_url()
            metadata = {
                "username": username,
                "full_name": full_name,
                "is_admin": is_admin,
            }
            auth_user = self.client.auth.sign_up({
                "email": email,
                "password": password,
                "options": {
                    "data": metadata,
                    "email_redirect_to": f"{app_base_url}/auth/callback?flow=confirm",
                },
            }).user

            if not auth_user:
                return {"success": False, "error": "Failed to create auth user"}

            profile_result = self.sync_user_profile(
                user_id=auth_user.id,
                email=email,
                username=username,
                full_name=full_name,
                is_admin=is_admin,
            )
            if not profile_result.get("success"):
                return profile_result

            return {"success": True, "user": auth_user, "profile": profile_result["profile"]}
        except Exception as exc:
            logger.error("Failed to register auth user: %s", exc)
            return {"success": False, "error": str(exc)}

    def login_user(self, email: str, password: str) -> Dict:
        if not self.client:
            return {"success": False, "error": "Supabase client not initialized"}

        try:
            response = self.client.auth.sign_in_with_password({"email": email, "password": password})
            if not response.user or not response.session:
                return {"success": False, "error": "Invalid credentials"}

            self.sync_user_profile(
                user_id=response.user.id,
                email=response.user.email or email,
                username=response.user.user_metadata.get("username") or email.split("@")[0],
                full_name=response.user.user_metadata.get("full_name"),
                is_admin=bool(response.user.user_metadata.get("is_admin", False)),
                last_login=utc_now().isoformat(),
            )

            return {"success": True, "user": response.user, "session": response.session}
        except Exception as exc:
            logger.error("Login failed: %s", exc)
            return {"success": False, "error": str(exc)}

    def get_auth_user(self, token: str) -> Dict:
        if not self.client:
            return {"success": False, "error": "Supabase client not initialized"}

        try:
            response = self.client.auth.get_user(token)
            if response and response.user:
                return {"success": True, "user": response.user}
            return {"success": False, "error": "Invalid token"}
        except Exception as exc:
            logger.error("Failed to validate auth token: %s", exc)
            return {"success": False, "error": str(exc)}

    def update_auth_user(
        self,
        user_id: str,
        email: str = None,
        username: str = None,
        full_name: str = None,
        is_admin: bool = None,
    ) -> Dict:
        if not self.client:
            return {"success": False, "error": "Supabase client not initialized"}
        if not self.admin_client:
            return {"success": False, "error": "Supabase service role key is required for profile updates"}

        try:
            auth_attributes = {}
            metadata = {}
            if email:
                auth_attributes["email"] = email
            if username is not None:
                metadata["username"] = username
            if full_name is not None:
                metadata["full_name"] = full_name
            if is_admin is not None:
                metadata["is_admin"] = is_admin
            if metadata:
                auth_attributes["user_metadata"] = metadata

            if auth_attributes:
                auth_user = self.admin_client.auth.admin.update_user_by_id(user_id, auth_attributes).user
            else:
                auth_user = self.admin_client.auth.admin.get_user_by_id(user_id).user

            current_profile_result = self.get_user_profile(user_id)
            current_profile = current_profile_result.get("profile", {}) if current_profile_result.get("success") else {}

            profile_result = self.sync_user_profile(
                user_id=user_id,
                email=email or auth_user.email or current_profile.get("email"),
                username=username or current_profile.get("username") or auth_user.user_metadata.get("username") or (auth_user.email.split("@")[0] if auth_user.email else user_id),
                full_name=full_name if full_name is not None else current_profile.get("full_name") or auth_user.user_metadata.get("full_name"),
                is_admin=is_admin if is_admin is not None else current_profile.get("is_admin", False),
                avatar_url=current_profile.get("avatar_url"),
            )
            if not profile_result.get("success"):
                return profile_result

            return {"success": True, "user": auth_user, "profile": profile_result["profile"]}
        except Exception as exc:
            logger.error("Failed to update auth user: %s", exc)
            return {"success": False, "error": str(exc)}

    def send_password_reset_email(self, email: str, redirect_to: Optional[str] = None) -> Dict:
        if not self.client:
            return {"success": False, "error": "Supabase client not initialized"}

        try:
            app_base_url = get_app_base_url()
            reset_redirect = redirect_to or f"{app_base_url}/auth/callback?flow=recovery"
            self.client.auth.reset_password_email(
                email,
                {
                    "redirect_to": reset_redirect,
                },
            )
            return {"success": True}
        except Exception as exc:
            logger.error("Failed to send password reset email: %s", exc)
            return {"success": False, "error": str(exc)}

    def reset_password_with_access_token(self, access_token: str, new_password: str) -> Dict:
        if not self.client:
            return {"success": False, "error": "Supabase client not initialized"}

        try:
            # Validate the recovery token and resolve the corresponding user.
            auth_result = self.get_auth_user(access_token)
            if not auth_result.get("success") or not auth_result.get("user"):
                return {"success": False, "error": "Reset link is invalid or expired"}

            user_id = auth_result["user"].id

            # Prefer service-role update for reliability across recovery token variations.
            if self.admin_client:
                self.admin_client.auth.admin.update_user_by_id(user_id, {"password": new_password})
                return {"success": True}

            # Fallback to user-scoped auth endpoint when service-role key is not configured.
            response = httpx.put(
                f"{self.url}/auth/v1/user",
                headers={
                    "apikey": self.key,
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
                json={"password": new_password},
                timeout=20.0,
            )

            if response.status_code >= 400:
                try:
                    payload = response.json()
                    error_message = payload.get("msg") or payload.get("error_description") or payload.get("error")
                except Exception:
                    error_message = response.text
                return {"success": False, "error": error_message or "Password reset failed"}

            return {"success": True}
        except Exception as exc:
            logger.error("Failed to reset password: %s", exc)
            return {"success": False, "error": str(exc)}

    def create_feedback(
        self,
        message: str,
        email: Optional[str] = None,
        name: Optional[str] = None,
        category: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict:
        table_client = self._server_table_client()
        if not table_client:
            return {"success": False, "error": "Supabase client not initialized"}

        payload = {
            "message": message,
            "email": email,
            "name": name,
            "category": category,
            "user_id": user_id,
            "status": "new",
        }

        try:
            response = table_client.table("feedback").insert(payload).execute()
            if not response.data:
                return {"success": False, "error": "Feedback was not stored"}
            return {"success": True, "feedback": response.data[0]}
        except Exception as exc:
            logger.error("Failed to store feedback: %s", exc)
            return {"success": False, "error": str(exc)}

    def get_feedback(self, page: int = 1, per_page: int = 20, status: Optional[str] = None) -> Dict:
        table_client = self._server_table_client()
        if not table_client:
            return {"success": False, "error": "Supabase client not initialized"}

        start = max(page - 1, 0) * per_page
        end = start + per_page - 1

        try:
            query = table_client.table("feedback").select("*", count="exact")
            if status and status.lower() != "all":
                query = query.eq("status", status)

            response = query.order("created_at", desc=True).range(start, end).execute()
            total = response.count or 0
            return {
                "success": True,
                "feedback": response.data or [],
                "total": total,
                "pages": (total + per_page - 1) // per_page if per_page else 1,
                "current_page": page,
            }
        except Exception as exc:
            logger.error("Failed to fetch feedback: %s", exc)
            return {"success": False, "error": str(exc)}

    def get_feedback_by_id(self, feedback_id: int) -> Dict:
        table_client = self._server_table_client()
        if not table_client:
            return {"success": False, "error": "Supabase client not initialized"}

        try:
            response = table_client.table("feedback").select("*").eq("id", feedback_id).limit(1).execute()
            if not response.data:
                return {"success": False, "error": "Feedback not found"}
            return {"success": True, "feedback": response.data[0]}
        except Exception as exc:
            logger.error("Failed to fetch feedback item: %s", exc)
            return {"success": False, "error": str(exc)}

    def reply_feedback(self, feedback_id: int, admin_response: str, responded_by: str) -> Dict:
        table_client = self._server_table_client()
        if not table_client:
            return {"success": False, "error": "Supabase client not initialized"}

        payload = {
            "status": "replied",
            "admin_response": admin_response,
            "responded_by": responded_by,
            "responded_at": utc_now().isoformat(),
        }

        try:
            response = table_client.table("feedback").update(payload).eq("id", feedback_id).execute()
            if not response.data:
                return {"success": False, "error": "Feedback not found"}
            return {"success": True, "feedback": response.data[0]}
        except Exception as exc:
            logger.error("Failed to reply to feedback: %s", exc)
            return {"success": False, "error": str(exc)}

supabase_client = SupabaseClient()
