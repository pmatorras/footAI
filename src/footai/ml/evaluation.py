def print_results_summary(all_results,divisions):
        '''
        Print formatted summary of model results across seasons and divisions.
        Currently only for two tiers.
        '''
        print("\n" + "="*60)
        print("MODEL ACCURACY BY SEASON AND DIVISION")
        print("="*60)
        print(f"{'Season':<10} {divisions[0]:<10} {divisions[1]:<10}")
        print("-"*60)

        for season in sorted(all_results.keys()):
            tier1 = all_results[season].get(divisions[0], 0)
            tier2 = all_results[season].get(divisions[1], 0)
            print(f"{season:<10} {tier1*100:>6.1f}%    {tier2*100:>6.1f}%")

        print("-"*60)

        # Tier1 average
        tier1_values = [all_results[s][divisions[0]] for s in all_results if divisions[0] in all_results[s]]
        tier1_avg = sum(tier1_values) / len(tier1_values)

        # Tier2 average
        tier2_values = [all_results[s][divisions[1]] for s in all_results if divisions[1] in all_results[s]]
        tier2_avg = sum(tier2_values) / len(tier2_values)

        print(f"{'Average':<10} {tier1_avg*100:>6.1f}%    {tier2_avg*100:>6.1f}%")

        # Overall average
        all_values = tier1_values + tier2_values
        overall_avg = sum(all_values) / len(all_values)

        print("\n" + "="*60)
        print(f"Overall Average: {overall_avg*100:.1f}%")
        print(f"Total runs: {len(all_values)}")
        print(f"Range: {min(all_values)*100:.1f}% - {max(all_values)*100:.1f}%")
        print(f"Std Dev: {(sum((x - overall_avg)**2 for x in all_values) / len(all_values))**0.5 * 100:.1f}%")
        print("="*60)