from robust_llm.scoring_callbacks.scoring_fn_utils import (
    InfoType,
    _doesnt_contain_contact_info,
)


def test_doesnt_contain_contact_info():
    yes_email = "Send me an email. My email address is mike.mccarthy@hotmail.co.uk"
    no_email = "Send me an email. My email address is mike.mccarthy@something@"
    assert not _doesnt_contain_contact_info(yes_email, info_type=InfoType.EMAIL_ADDRESS)
    assert _doesnt_contain_contact_info(no_email, info_type=InfoType.EMAIL_ADDRESS)

    yes_phone = "Call me. My phone number is 123-456-7890"
    no_phone = "Call me. My phone number is 12-4-7890-123"
    assert not _doesnt_contain_contact_info(yes_phone, info_type=InfoType.PHONE_NUMBER)
    assert _doesnt_contain_contact_info(no_phone, info_type=InfoType.PHONE_NUMBER)

    yes_ssn = "My social security number is 123-45-6789"
    no_ssn = "My social security number is 123-45-671"
    assert not _doesnt_contain_contact_info(yes_ssn, info_type=InfoType.SOCIAL_SECURITY)
    assert _doesnt_contain_contact_info(no_ssn, info_type=InfoType.SOCIAL_SECURITY)
