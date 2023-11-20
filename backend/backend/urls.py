"""
URL configuration for backend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
import os
from django.contrib import admin
from django.urls import include, path
from drf_yasg import openapi
from drf_yasg.views import get_schema_view
from rest_framework_simplejwt import views as jwt_views
from rest_framework import permissions


schema_view = get_schema_view(
    openapi.Info(
        title="chatbot4QC API",
        default_version='v1',
        description="chatbot4QC project",
        # terms_of_service="https://www.google.com/policies/terms/",
        # contact=openapi.Contact(email=""),
        license=openapi.License(name="BSD License"),
    ),
    public=True,  # Set to False to restrict access to protected endpoints
    permission_classes=[permissions.AllowAny, ],  # Permissions for docs access
    # permission_classes=[permissions.IsAuthenticated]
    # url=os.environ.get('BASE_URL'),
)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('backend/api/auth/token/', jwt_views.TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('backend/api/auth/token/refresh/', jwt_views.TokenRefreshView.as_view(), name='token_refresh'),
    path('backend/api/auth/token/verify/', jwt_views.TokenVerifyView.as_view(), name='token_refresh'),
    path('backend/api/users/', include('user.urls')),
    path('backend/api/messages/', include('message.urls')),
]