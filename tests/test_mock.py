from jevpoc.mock import MockJev

QS = {
    "c": {"type": "choice", "instructions": "x", "criteria": {"billing": "bill", "technical": "error"}},
    "s": {"type": "score", "instructions": "x", "criteria": ["calm", "angry"]},
    "n": {"type": "noul", "instructions": "x"},
}


def test_shapes_and_ranges():
    r = MockJev("m").ask({"t": "HTTP 500 error"}, QS)
    assert r.answers["c"]["choice"] in ("billing", "technical")
    assert abs(sum(r.answers["c"]["probabilities"].values()) - 1) < 0.01
    assert 0 <= r.score("s") <= 1
    assert 0 <= r.noul("n") <= 1 and "confidence" not in r.answers["n"]
    assert r.backend == "mock"


def test_deterministic_for_same_request():
    a = MockJev("m").ask({"t": "x"}, QS).answers
    b = MockJev("m").ask({"t": "x"}, QS).answers
    assert a == b
