#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""
from hw1.params import Arguments
from hw1.models import run_model
from hw1 import load_policy
from hw1 import behavioral_cloning


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    args = Arguments()
    parser.add_argument('--expert_policy_file',
                        type=str,
                        default=args.get_model_file()
                        )
    parser.add_argument('--env_name',
                        type=str,
                        default=args.env_name
                        )
    parser.add_argument('--render',
                        action='store_true',
                        default=args.render
                        )
    parser.add_argument("--max_timesteps",
                        type=int,
                        default=args.max_timesteps
                        )
    parser.add_argument('--num_rollouts',
                        type=int,
                        default=args.rollouts,
                        help='Number of expert rollouts / episodes.'
                        )
    parser.add_argument('--hidden_units',
                        type=int,
                        default=args.hidden_units
                        )
    parser.add_argument('--model_type',
                       type=str,
                       default=args.model_type)
    parser.add_argument('--expert_data_dir',
                        type=str,
                        default=args.expert_data_dir)
    parser.add_argument('--verbose',
                        type=bool,
                        default=args.verbose)


    parsed_args = parser.parse_args()
    args.set_parsed_args(parsed_args=parsed_args)
    return args


if __name__ == '__main__':
    args = get_args()
    print('loading and building expert policy')
    if str(args.expert_policy_file).startswith("behave_models"):
        policy_fn = behavioral_cloning.load_policy(args=args)
    else:
        policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')
    args.rollouts = 100
    run_model(args=args, policy_fn=policy_fn)
