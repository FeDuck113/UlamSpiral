from nums import PSEUDOPRIME_NUMS
import math
import numpy
import pandas as pd
from typing import Literal, Tuple, List
from pydantic import BaseModel, Field


class NumInfo(BaseModel):
    num: int = Field(ge=0)
    x_coord: int
    y_coord: int
    radius: float
    angle: float
    label: Literal["composite", "prime", "pseudoprime"]
    pseudoprime_types: List[
        Literal[
            "CARMICHAEL",
            "FERMAT_2",
            "FERMAT_3",
            "FERMAT_5",
            "MILLER_RABIN_2",
            "MILLER_RABIN_3",
            "MILLER_RABIN_5",
            "EULER_2",
            "EULER_3",
            "EULER_5",
            "ABSOLUTE_EULER",
            "EULER_JACOBI_2",
            "EULER_JACOBI_3",
            "EULER_JACOBI_5",
            "BRUCKMAN_LUCAS",
            "ODD_FIBONACCI",
            "UNRESTRICTED_PERRIN",
            "RESTRICTED_PERRIN",
            "NSW",
            "FROBENIUS",
            "CATALAN",
        ]
    ] = []


def sieve_of_eratosthenes(n: int):
    primes = [True] * (n + 1)
    for p in range(2, int(n**0.5) + 1):
        if primes[p]:
            for i in range(p * p, n + 1, p):
                primes[i] = False
    return [p for p, is_prime in enumerate(primes) if is_prime]


def check_pseudoprime(
    num: int,
) -> Tuple[Literal["composite", "pseudoprime"], List[str]]:
    types = []
    for type_name, nums in PSEUDOPRIME_NUMS.items():
        if num in nums:
            types.append(type_name)
    status = "pseudoprime" if types else "composite"
    return (status, types)


def get_num_info(num: int, x_coord: int, y_coord: int) -> NumInfo:
    global primes

    radius = (x_coord**2 + y_coord**2) ** 0.5
    angle = math.atan2(y_coord, x_coord) % (2 * numpy.pi)

    label = "composite"
    pseudoprime_types = []

    if num in primes:
        label = "prime"
    else:
        label, pseudoprime_types = check_pseudoprime(num)

    return NumInfo(
        num=num,
        x_coord=x_coord,
        y_coord=y_coord,
        radius=radius,
        angle=angle,
        label=label,
        pseudoprime_types=pseudoprime_types,
    )


def generate_spiral(N):
    global primes

    x, y = 0, 0

    dx, dy = 1, 0
    step_count = 0
    turn_step_count = 1
    turn_count = 0

    data = []

    primes = sieve_of_eratosthenes(N)

    for num in range(1, N + 1):
        print(num)
        data.append(get_num_info(num, x, y).model_dump())

        x += dx
        y += dy
        step_count += 1

        if step_count == turn_step_count:
            step_count = 0

            dx, dy = -dy, dx

            if turn_count % 2 == 0:
                turn_step_count += 1

    return data


def create_spiral_dataframe(N):
    df = pd.DataFrame(generate_spiral(N), columns=NumInfo.__fields__.keys())
    print(df.head())
    df.to_csv(f"{N}-spiral.csv")


if __name__ == "__main__":
    create_spiral_dataframe(1000000)
