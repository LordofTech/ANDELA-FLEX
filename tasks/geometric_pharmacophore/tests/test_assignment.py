from tasks.geometric_pharmacophore.assignment import capped_product, translate_only


def test_capped_product():
    opts = [[1, 2], ["a", "b"]]
    got = list(capped_product(opts, 3))
    assert len(got) == 3


def test_translate_only():
    t = translate_only((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert t == (1.0, -1.0, 0.0)
