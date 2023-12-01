from rest_framework.generics import ListCreateAPIView, RetrieveUpdateDestroyAPIView

from message.models import Message
from message.serializers import MessageSerializer

from user.models import User




class ListCreateMessageView(ListCreateAPIView):
    """
    get: Lists all messages of the logged-in in chronological order.

    post: Creates a new Message.
    """
    serializer_class = MessageSerializer

    def get_queryset(self):
        return Message.objects.order_by("updated")

    def perform_create(self, serializer):

        # For now everything is associated to the same user, which is the only one.
        user = User.objects.get(email='user@email.com')
        serializer.save(user=user)


class RetrieveUpdateDeleteMessageView(RetrieveUpdateDestroyAPIView):
    """
        get: Retrieves a specific Message.
        patch: Updates a specific Message.
        delete: Deletes a Message.
    """
    queryset = Message.objects.all()
    serializer_class = MessageSerializer
    lookup_field = 'id'  # field in the database
    lookup_url_kwarg = 'id_message'  # field in the request
    http_method_names = ['get', 'patch', 'delete']  # disallow put as we don't use it


