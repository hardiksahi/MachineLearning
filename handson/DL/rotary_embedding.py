import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_rotation_matrix(abs_position, theta):
    rotation_matrix = [
        [
            np.cos(np.deg2rad(abs_position * theta)),
            -np.sin(np.deg2rad(abs_position * theta)),
        ],
        [
            np.sin(np.deg2rad(abs_position * theta)),
            np.cos(np.deg2rad(abs_position * theta)),
        ],
    ]

    return rotation_matrix


def plot_absolute_embedding(original_vector, rotated_vector_list, token):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.quiver(
        0,
        0,
        original_vector[0],
        original_vector[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="red",
    )
    ax.annotate(
        "Original",
        xy=(original_vector[0], original_vector[1]),
    )
    for abs_pos, rotated_vector in enumerate(rotated_vector_list):
        ax.quiver(
            0,
            0,
            rotated_vector[0],
            rotated_vector[1],
            angles="xy",
            scale_units="xy",
            scale=1,
        )
        ax.annotate(
            f"Pos: {abs_pos+1}",
            xy=(rotated_vector[0], rotated_vector[1]),
        )
    ax.set_xlim([-12, 12])
    ax.set_ylim([-12, 12])
    ax.set_title(f"Absolute position embedding for : {token}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    return fig


def plot_relative_embedding(token_pos_rotated_vector_dict):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for token, inner_dict in token_pos_rotated_vector_dict.items():
        position = inner_dict["position"]
        shifted_position = inner_dict["shifted_position"]
        vector = inner_dict["vector"]
        shifted_vector = inner_dict["shifted_vector"]

        ax.quiver(
            0,
            0,
            vector[0],
            vector[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="red",
        )
        ax.annotate(
            f"{token}, pos:{position}",
            xy=(vector[0], vector[1]),
        )

        ax.quiver(
            0,
            0,
            shifted_vector[0],
            shifted_vector[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="blue",
        )
        ax.annotate(
            f"{token}, pos:{shifted_position}",
            xy=(shifted_vector[0], shifted_vector[1]),
        )

    ax.set_xlim([-12, 12])
    ax.set_ylim([-12, 12])
    ax.set_title("Relative position embedding")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    return fig


if __name__ == "__main__":
    ## COnsider 2D vector only
    dimensions = 2
    seq_len = 10
    dog = np.random.randint(2, 9, dimensions)
    theta = 30  ##(fixed/ Hardcoded)

    ## Step 1: Understand absolute embedding
    dog_abs_pos_matrix_dict = {
        i: get_rotation_matrix(i, theta) for i in range(1, seq_len + 1)
    }
    # print(abs_pos_matrix_dict_t1)
    dog_rotated_vector_list = [
        np.dot(dog_abs_pos_matrix_dict[i], dog) for i in range(1, seq_len + 1)
    ]

    ## Step 2: Plor original and rotated vectors
    dog_plot = plot_absolute_embedding(dog, dog_rotated_vector_list, token="dog")
    dog_plot.savefig("Absolute_position_ROPA.png")

    ## RoPA ensures that the absolute position embedding of a token is simplyt rotation by abs_position* theta

    ## Step 2: Understand relative embedding
    cat = np.random.randint(2, 9, dimensions)

    pos_dog = np.random.choice(np.arange(1, seq_len + 1), 1)[0]
    pos_cat = np.random.choice(
        [element for element in np.arange(1, seq_len + 1) if element != pos_dog], 1
    )[0]

    print(f"Cat and dog are separated by: {abs(pos_dog-pos_cat)} positions")

    dog_rotation_matrix = get_rotation_matrix(pos_dog, theta)
    dog_rotated_vector = np.dot(dog_rotation_matrix, dog)

    cat_rotation_matrix = get_rotation_matrix(pos_cat, theta)
    cat_rotated_vector = np.dot(cat_rotation_matrix, cat)

    shift_by = 10
    shifted_pos_dog = pos_dog + shift_by
    shifted_pos_cat = pos_cat + shift_by

    shifted_dog_rotation_matrix = get_rotation_matrix(shifted_pos_dog, theta)
    shifted_dog_rotated_vector = np.dot(shifted_dog_rotation_matrix, dog)

    shifted_cat_rotation_matrix = get_rotation_matrix(shifted_pos_cat, theta)
    shifted_cat_rotated_vector = np.dot(shifted_cat_rotation_matrix, cat)

    relative_plot = plot_relative_embedding(
        {
            "dog": {
                "position": pos_dog,
                "vector": dog_rotated_vector,
                "shifted_vector": shifted_dog_rotated_vector,
                "shifted_position": shifted_pos_dog,
            },
            "cat": {
                "position": pos_cat,
                "vector": cat_rotated_vector,
                "shifted_vector": shifted_cat_rotated_vector,
                "shifted_position": shifted_pos_cat,
            },
        }
    )

    relative_plot.savefig("Relative_position_ROPA.png")

    ## This shows that RoPA handles absolute position and relative position of 2 tokens effectively
