from django.contrib.auth import get_user_model
from django.db.models import Q
from rest_framework.generics import ListAPIView, RetrieveUpdateDestroyAPIView, RetrieveAPIView
from rest_framework.permissions import IsAuthenticated

# from message.models import Message
from user.serializers import UserSerializer


User = get_user_model()


class ListUserView(ListAPIView):
    """
    get:
    Lists all users.
    """
    serializer_class = UserSerializer
    # permission_classes = [IsAuthenticated]

    def get_queryset(self):
        query = self.request.GET.get('search', '')  # search is the params and '' the default value
        queryset = User.objects.filter(
            Q(email__contains=query) | Q(first_name__contains=query) | Q(last_name__contains=query))
        return queryset


class LoggedInUserView(RetrieveUpdateDestroyAPIView):
    """
        get:
        Retrieves logged-in User and displays all details.

        patch:
        Updates details of logged-in User.

        delete:
        Deletes logged-in User.

    """
    http_method_names = ['get', 'patch', 'delete']  # disallow put as we don't use it
    serializer_class = UserSerializer
    # permission_classes = [IsAuthenticated]

    def get_object(self):
        return self.request.user


class RetrieveUserView(RetrieveAPIView):
    """
        get:
        Retrieves a specific User by ID and displays all the information about it.

        patch:
        Updates the status of a specific User.

        delete:
        Deletes a User by ID.

    """
    lookup_field = 'id'
    lookup_url_kwarg = 'id_user'
    queryset = User.objects.all()
    serializer_class = UserSerializer
    # permission_classes = [IsAuthenticated]
