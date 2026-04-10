cana_map = {
    # necessity
    (-1, 0): (1, 0),
    (1, 0): (0, 0),
    # sufficiency
    (-1, 1): (0, 1),
    (1, 1): (1, 1),
}


def cana_tax_to_values(cana4):
    return cana4[:2] + cana_map(cana4[2:])
