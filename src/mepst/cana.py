cana_map = {
    # necessity
    (-1, 0): (1, 0),
    (1, 0): (0, 0),
    # sufficiency
    (-1, 1): (0, 1),
    (1, 1): (1, 1),
}


def cana_tax_to_values(cana4):
    return cana4[:2] + cana_map[cana4[2:]]


def cana_values_to_sign(cana4):
    if cana4[-1] == cana4[-2]:
        return (*cana4[:2], 1)
    return (*cana4[:2], -1)
