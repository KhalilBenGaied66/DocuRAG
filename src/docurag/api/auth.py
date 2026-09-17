"""Identity comes from the calling application's token; groups drive document access."""

from __future__ import annotations

import asyncio

import jwt
from fastapi import Depends, Request

from docurag.api.errors import ApiError
from docurag.config import Settings
from docurag.models import User

_BEARER = {"WWW-Authenticate": "Bearer"}


class JWTVerifier:
    """Validates Entra ID (or any OIDC) access tokens against the issuer's published keys."""

    def __init__(self, settings: Settings) -> None:
        if not (settings.jwt_jwks_url and settings.jwt_issuer and settings.jwt_audience):
            raise RuntimeError(
                "auth_mode=jwt needs DOCURAG_JWT_JWKS_URL, DOCURAG_JWT_ISSUER, DOCURAG_JWT_AUDIENCE"
            )
        self.settings = settings
        self._jwks = jwt.PyJWKClient(settings.jwt_jwks_url, cache_keys=True, lifespan=3600)

    def verify(self, token: str) -> User:
        s = self.settings
        key = self._jwks.get_signing_key_from_jwt(token).key
        claims = jwt.decode(
            token, key, algorithms=["RS256"], audience=s.jwt_audience, issuer=s.jwt_issuer,
            options={"require": ["exp", "iss", "aud"]},
        )
        if s.jwt_groups_claim not in claims and "_claim_names" in claims:
            # Entra ID "group overage": too many groups to fit in the token.
            raise ApiError(
                403, "group_overage",
                "the token lists no groups (overage); use app roles: DOCURAG_JWT_GROUPS_CLAIM=roles",
            )
        groups = claims.get(s.jwt_groups_claim) or []
        if isinstance(groups, str):
            groups = [groups]
        user_id = claims.get("oid") or claims.get("sub")
        if not user_id:
            raise ApiError(401, "invalid_token", "token has no subject", _BEARER)
        return User(id=str(user_id), groups=[str(g) for g in groups])


async def current_user(request: Request) -> User:
    settings: Settings = request.app.state.container.settings
    if settings.auth_mode == "dev":
        user_id = request.headers.get("x-dev-user", "").strip()
        groups = [g.strip() for g in request.headers.get("x-dev-groups", "").split(",") if g.strip()]
        if not user_id or not groups:
            raise ApiError(401, "unauthenticated", "dev mode: send X-Dev-User and X-Dev-Groups")
        return User(id=user_id, groups=groups)

    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise ApiError(401, "unauthenticated", "missing bearer token", _BEARER)
    verifier = getattr(request.app.state, "jwt_verifier", None)
    if verifier is None:
        verifier = request.app.state.jwt_verifier = JWTVerifier(settings)
    try:
        # Key fetching is blocking I/O (cached after the first call).
        return await asyncio.to_thread(verifier.verify, token)
    except jwt.PyJWTError as exc:
        raise ApiError(401, "invalid_token", f"invalid token: {exc}", _BEARER) from exc


async def require_admin(request: Request, user: User = Depends(current_user)) -> User:
    if request.app.state.container.settings.admin_group not in user.groups:
        raise ApiError(403, "forbidden", "document management needs the admin group")
    return user
