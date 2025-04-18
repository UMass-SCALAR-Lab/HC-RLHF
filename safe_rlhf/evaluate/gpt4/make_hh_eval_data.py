from datasets import load_dataset, concatenate_datasets
import json
import argparse, sys

if __name__ == '__main__':

    CACHE_DIR = "/scratch3/workspace/ychittepu_umass_edu-cql/cache/"
    TOTAL_DATA_POINTS = 100

    inp_args = {
        'dataset': 'PKU-Alignment/PKU-SafeRLHF',
        'seed': 123,
        'out_file': "hh_prompts.json",
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        help='the name of the dataset',
        default=None,
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='the seed',
    )
    parser.add_argument(
        '--out_file',
        type=str,
        help='the name of the output json file',
        default=None,
    )
    args = parser.parse_args()

    for arg, value in vars(args).items():
        if(value is not None):
            inp_args[arg] = value

    dset = load_dataset(inp_args['dataset'], cache_dir=CACHE_DIR, split='test')
    dset = dset.shuffle(seed=42)
    dset = dset.select(range(4000, dset.num_rows))

    # Define the boolean field combinations
    combinations = [(True, True), (True, False), (False, True), (False, False)]

    # proportions = [45, 25, 25, 25]
    proportions = [40, 20, 20, 20]

    # Initialize a list to store sampled subsets
    sampled_subsets = []

    for combo in combinations:
        # Extract field values
        is_resp_0, is_resp_1 = combo
        
        # Filter dataset based on current combination
        filtered_dataset = dset.filter(lambda x: x['is_response_0_safe'] == is_resp_0 and x['is_response_1_safe'] == is_resp_1)
        
        # Randomly select up to 25 samples if available
        sampled = filtered_dataset.shuffle(seed=inp_args['seed']).select(range(min(TOTAL_DATA_POINTS//len(combinations), len(filtered_dataset))))
        
        # Add to the final list
        sampled_subsets.append(sampled)

    final_dset = concatenate_datasets(sampled_subsets).shuffle()

    # Extract 'prompt' field and format it into the required JSON structure
    json_data = [{"prompt": entry["prompt"]} for entry in final_dset]

    # Convert to JSON string
    json_output = json.dumps(json_data, indent=4)

    # Save to a file (optional)
    with open(inp_args['out_file'], "w") as f:
        f.write(json_output)




