import argparse
import os
import yaml
from symmer.projection import QubitTapering
from symmer.operators import PauliwordOp
from symmer.projection import CS_VQE
import datetime

def check_path_to_dir(potential_path: str) -> str:
    """
    Checks if path is a directory

    Args:
        potential_path (str): path to directory

    Returns:
        potential_path (str): valid path (error raised if not)
    """
    if os.path.isdir(potential_path) is False:
        raise argparse.ArgumentTypeError("directory path defined is not valid")
    return potential_path


def check_path_to_file(potential_path: str) -> str:
    """
    Checks if path gives a file

    Args:
        potential_path (str): path to directory

    Returns:
        potential_path (str): valid path (error raised if not)
    """
    if os.path.isfile(potential_path) is False:
        raise argparse.ArgumentTypeError("file path defined is not valid")
    return potential_path


def command_interface():
    """
    Parse arguments from command line interface.

    test:
    python command_line.py taper --config ~/Documents/PhD/SymRed/tests/yaml_input/H2_JW_real.yaml
    """

    parser = argparse.ArgumentParser(description="Output directory.")
    #
    # # https://stackoverflow.com/questions/27529610/call-function-based-on-argparse
    # FUNCTION_MAP = { 'taper': None, #TODO: add taper function
    #                  'contextual_subspace': None #TODO: add CS function
    #
    # }
    #
    # parser.add_argument('command',
    #                     choices=FUNCTION_MAP.keys(),
    #                     type=str,
    #                     help="command of algorithm to implement (tapering or contextual subspace approx)")

    parser.add_argument('--command',
                        type=str,
                        help="command of algorithm to implement (tapering or contextual subspace approx)")

    parser.add_argument(
        "--config",
        type=str,
        help="Path to a YAML config file. Overwrites other arguments.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=check_path_to_dir,
        help="Path to a output directory (if none given then current working dir is used).",
    )
    parser.add_argument(
        "--Hamiltonian",
        '-H',
        # type=check_path_to_file,
        type=dict,
        help="Path to Pauli Hamiltonian (json file of Hamiltonian).", #TODO: add other options too
    )
    parser.add_argument(
        "--verbose",
        "-v",
        type=bool,
        help="Whether to print details of function applied",
    )

    parser.add_argument(
        "--taper_reference",
        "-tr",
        type=list,
        help="optional list of {0,1} representing slater determinant in correct symmetry sector",
    )
    parser.add_argument(
        "--contextual_subspace_enforce_clique_operator",
        '-enforce_A',
        type=bool,
        help="whether to enforce clique operator (A)",
    )
    args = parser.parse_args()
    if args.config:
        # "Reading config file."
        with open(args.config, "r") as infile:
            args = yaml.safe_load(infile)

            args["command"] = args.command

            # Set optional argument defaults
            args["verbose"] = args.get("verbose", False)
            args["output_dir"] = args.get("output_dir", os.getcwd())
            args["taper_reference"] = args.get("taper_reference", None)
            args["contextual_subspace_enforce_clique_operator"] = args.get("contextual_subspace_enforce_clique_operator",
                                                                           True)
    else:
        # Transform the namespace object to a dict.
        args = vars(args)
        args["command"] = args.command

    if any([values is None for values in args.values()]):
        print(
            f"Missing values for argument {[key for key, value in args.items() if value is None]}"
        )
        print("\nMissing values for arguments: ".upper())
        print(f"{[key for key, value in args.items() if value is None]}\n")
        raise Exception("Missing argument values.")


    return args


def cli() -> None:
    """
    command line interface
    """
    args = command_interface()

    output_data = {}

    if args["taper"] == 'taper':
        basename = 'tapering'
        print(args["H"])
        taper_hamiltonian = QubitTapering(PauliwordOp(args["H"]))
        reference_state = QubitTapering(PauliwordOp(args["taper_reference"]))

        taper_hamiltonian.stabilizers.rotate_onto_single_qubit_paulis()
        taper_hamiltonian.stabilizers.update_sector(reference_state)
        ham_tap = taper_hamiltonian.taper_it(ref_state=reference_state)

        output_data['tapered_H'] = ham_tap.to_dictionary
        output_data['symmetry_generators'] = taper_hamiltonian.symmetry_generators.to_dictionary
        output_data['symmetry_generators']= [rot.to_dictionary for rot in
                                             taper_hamiltonian.stabilizers.stabilizer_rotations]


    elif args["contextual_subspace"] == 'contextual_subspace':
        basename = 'contextual_subspace'

        cs_vqe = CS_VQE(PauliwordOp(args["H"]))
        noncon_H = cs_vqe.noncontextual_operator
        con_H = cs_vqe.contextual_operator

        noncon_symmetry_generators = cs_vqe.symmetry_generators
        clique_operator = cs_vqe.clique_operator


        output_data['noncon_H'] = noncon_H.to_dictionary
        output_data['con_H'] = con_H.to_dictionary
        output_data['noncon_symmetry_generators'] = noncon_symmetry_generators.to_dictionary
        output_data['clique_operator'] = clique_operator.to_dictionary

        #TODO solve contextual subspace problem to find right sector!
    else:
        raise ValueError('unknown function')

    # make filename unique with date and time
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = "_".join([basename, suffix])


    outloc = os.path.join(args["output_dir"], filename + '.yaml')
    with open(os.path.join(outloc, 'w')) as file:
        yaml.dump(output_data, file)
        print(f'file saved at: {outloc}')

    return None


if __name__ == "__main__":
    # run command line code
    cli()
