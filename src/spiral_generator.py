from nums import PSEUDOPRIME_NUMS
import math
import numpy
import os
import pandas as pd
from typing import Literal, Tuple, List
from pydantic import BaseModel, Field

PSEUDOPRIME_SETS = {name: set(nums) for name, nums in PSEUDOPRIME_NUMS.items()}


class NumInfo(BaseModel):
    num: int = Field(ge=0)
    x_coord: int
    y_coord: int
    radius: float
    angle_rad: float
    angle_deg: float
    label: Literal["composite", "prime", "pseudoprime"]
    is_pseudo: int
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
    return {p for p, is_prime in enumerate(primes) if is_prime and p >= 2}


def check_pseudoprime(num: int) -> Tuple[Literal["composite", "pseudoprime"], List[str]]:
    types = []
    for type_name, nums_set in PSEUDOPRIME_SETS.items():
        if num in nums_set:
            types.append(type_name)
    status = "pseudoprime" if types else "composite"
    return (status, types)


def get_num_info(num: int, x_coord: int, y_coord: int) -> NumInfo:
    global primes

    radius = (x_coord**2 + y_coord**2) ** 0.5
    angle_rad = math.atan2(y_coord, x_coord) % (2 * numpy.pi)
    angle_deg = math.degrees(angle_rad)

    label = "composite"
    pseudoprime_types = []
    is_pseudo = 0

    if num in primes:
        label = "prime"
    else:
        label, pseudoprime_types = check_pseudoprime(num)
        if label == "pseudoprime":
            is_pseudo = 1

    return NumInfo(
        num=num,
        x_coord=x_coord,
        y_coord=y_coord,
        radius=round(radius, 4),
        angle_rad=round(angle_rad, 4),
        angle_deg=round(angle_deg, 2),
        label=label,
        is_pseudo=is_pseudo,
        pseudoprime_types=pseudoprime_types
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
        if num % 100000 == 0:
            print(f"Сгенерировано {num} из {N}...")
        info_obj = get_num_info(num, x, y)
        data.append(info_obj.model_dump())

        x += dx
        y += dy
        step_count += 1

        if step_count == turn_step_count:
            step_count = 0
            dx, dy = -dy, dx
            turn_count += 1
            if turn_count % 2 == 0:
                turn_step_count += 1

    return data


def create_spiral_dataframe(N):
    data = generate_spiral(N)
    df = pd.DataFrame(data, columns=NumInfo.model_fields.keys())
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    file_path = os.path.join(data_dir, f"{N}-spiral.csv")
    print(f"Сохранение в {file_path}...")
    df.to_csv(file_path, index=False)
    print("Файл успешно сохранен!")


if __name__ == "__main__":
    create_spiral_dataframe(1000000)
