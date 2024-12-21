from django.contrib.auth.forms import UserChangeForm
from django.contrib.auth import get_user_model
from django.urls import reverse

class CustomUserChangeForm(UserChangeForm):
    # password = None
    class Meta:
        model = get_user_model() # 현재 활성화된 유저 접근
        fields = ["first_name", "last_name", "email"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if self.fields.get("password"):
            password_help_text = (
                "You can change the password " '<a href="{}">here</a>.'
            ).format(f"{reverse('accounts:change_password')}")
            self.fields["password"].help_text = password_help_text