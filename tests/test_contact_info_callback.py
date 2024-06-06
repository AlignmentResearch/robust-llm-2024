from robust_llm.scoring_callbacks import doesnt_contain_contact_info


def test_doesnt_contain_contact_info():
    yes_email = "Send me an email. My email address is mike.mccarthy@hotmail.co.uk"
    no_email = "Send me an email. My email address is mike.mccarthy@something@"
    assert not doesnt_contain_contact_info(yes_email)
    assert doesnt_contain_contact_info(no_email)

    yes_phone = "Call me. My phone number is 123-456-7890"
    no_phone = "Call me. My phone number is 12-4-7890-123"
    assert not doesnt_contain_contact_info(yes_phone)
    assert doesnt_contain_contact_info(no_phone)

    yes_ssn = "My social security number is 123-45-6789"
    no_ssn = "My social security number is 123-45-671"
    assert not doesnt_contain_contact_info(yes_ssn)
    assert doesnt_contain_contact_info(no_ssn)
