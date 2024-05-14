import pytest


@pytest.fixture(scope="module")
def flash_omeganeo_handle(launcher):
    with launcher("qxsecureserver/omega-neo-xlarge-chat-awq") as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_omeganeo(flash_omeganeo_handle):
    await flash_omeganeo_handle.health(300)
    return flash_omeganeo_handle.client


@pytest.mark.asyncio
async def test_flash_omeganeo(flash_omeganeo, response_snapshot):
    response = await flash_omeganeo.generate(
        "Test request", max_new_tokens=10, decoder_input_details=True
    )

    assert response.details.generated_tokens == 10
    assert response.generated_text == "\n# Create a request\nrequest = requests.get"
    assert response == response_snapshot


@pytest.mark.asyncio
async def test_flash_omeganeo_all_params(flash_omeganeo, response_snapshot):
    response = await flash_omeganeo.generate(
        "Test request",
        max_new_tokens=10,
        repetition_penalty=1.2,
        return_full_text=True,
        stop_sequences=["test"],
        temperature=0.5,
        top_p=0.9,
        top_k=10,
        truncate=5,
        typical_p=0.9,
        watermark=True,
        decoder_input_details=True,
        seed=0,
    )

    assert response.details.generated_tokens == 10
    assert response == response_snapshot


@pytest.mark.asyncio
async def test_flash_omeganeo_load(flash_omeganeo, generate_load, response_snapshot):
    responses = await generate_load(flash_omeganeo, "Test request", max_new_tokens=10, n=4)

    assert len(responses) == 4
    assert all(
        [r.generated_text == responses[0].generated_text for r in responses]
    ), f"{[r.generated_text  for r in responses]}"
    assert responses[0].generated_text == "\n# Create a request\nrequest = requests.get"

    assert responses == response_snapshot
