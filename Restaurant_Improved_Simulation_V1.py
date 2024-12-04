import random
import math
import matplotlib.pyplot as plt
import numpy as np


class Restaurant:
    def __init__(self, name, quality):
        self.name = name
        self.quality = quality


def generate_private_signal(true_quality):
    # Generate a more diverse private signal (0 to 1)
    noise = random.gauss(0, 0.2)  # Gaussian noise
    signal = true_quality + noise
    return max(0, min(1, signal))  # Clamp between 0 and 1


def calculate_posterior(prior, signal, quality):
    # Calculate posterior probability using Bayes' rule
    likelihood = 1 - abs(signal - quality)  # Simplified likelihood
    return (prior * likelihood) / ((prior * likelihood) + ((1 - prior) * (1 - likelihood)))


def make_decision(private_signal, previous_choices, restaurant_a, restaurant_b, independence_factor, delay):
    prior_a = 0.5  # Initial prior probability for restaurant A

    # Update prior based on previous choices, but with less weight
    for choice in previous_choices[-delay:]:  # Consider only recent choices due to delay
        if choice == 'A':
            prior_a = calculate_posterior(prior_a, 0.6, restaurant_a.quality)
        else:
            prior_a = calculate_posterior(prior_a, 0.4, restaurant_a.quality)

    # Incorporate private signal with more weight (independence)
    posterior_a = calculate_posterior(prior_a, private_signal, restaurant_a.quality)

    # Adjust posterior based on independence factor
    adjusted_posterior_a = (independence_factor * private_signal) + ((1 - independence_factor) * posterior_a)

    return 'A' if adjusted_posterior_a > 0.5 else 'B'


def run_simulation(num_people, restaurant_a, restaurant_b, independence_factor, delay):
    choices = []
    for _ in range(num_people):
        private_signal = generate_private_signal(restaurant_a.quality)
        choice = make_decision(private_signal, choices, restaurant_a, restaurant_b, independence_factor, delay)
        choices.append(choice)

    return choices


def analyze_results(choices):
    correct_choices = choices.count('A')
    accuracy = correct_choices / len(choices)

    cascade_start = None
    for i in range(2, len(choices)):
        if choices[i] == choices[i - 1] == choices[i - 2]:
            cascade_start = i - 1
            break

    return accuracy, cascade_start


# Run multiple simulations
num_simulations = 1000
num_people = 20
restaurant_a = Restaurant('A', 0.6)  # Slightly better
restaurant_b = Restaurant('B', 0.4)
independence_factors = [0.1, 0.3, 0.5, 0.7, 0.9]
delays = [1, 3, 5, 7, 9]

results = {}

for independence_factor in independence_factors:
    for delay in delays:
        total_accuracy = 0
        cascade_count = 0
        all_accuracies = []
        cascade_starts = []

        for _ in range(num_simulations):
            choices = run_simulation(num_people, restaurant_a, restaurant_b, independence_factor, delay)
            accuracy, cascade_start = analyze_results(choices)
            total_accuracy += accuracy
            all_accuracies.append(accuracy)
            if cascade_start is not None:
                cascade_count += 1
                cascade_starts.append(cascade_start)

        avg_accuracy = total_accuracy / num_simulations
        cascade_probability = cascade_count / num_simulations

        results[(independence_factor, delay)] = {
            'avg_accuracy': avg_accuracy,
            'cascade_probability': cascade_probability,
            'all_accuracies': all_accuracies,
            'cascade_starts': cascade_starts
        }

# Plotting
plt.figure(figsize=(15, 10))

# 1. Heatmap of average accuracy
plt.subplot(2, 2, 1)
accuracy_data = [[results[(i, d)]['avg_accuracy'] for d in delays] for i in independence_factors]
plt.imshow(accuracy_data, cmap='YlOrRd', aspect='auto')
plt.colorbar(label='Average Accuracy')
plt.title('Average Accuracy')
plt.xlabel('Delay')
plt.ylabel('Independence Factor')
plt.xticks(range(len(delays)), delays)
plt.yticks(range(len(independence_factors)), independence_factors)

# 2. Heatmap of cascade probability
plt.subplot(2, 2, 2)
cascade_data = [[results[(i, d)]['cascade_probability'] for d in delays] for i in independence_factors]
plt.imshow(cascade_data, cmap='YlOrRd', aspect='auto')
plt.colorbar(label='Cascade Probability')
plt.title('Cascade Probability')
plt.xlabel('Delay')
plt.ylabel('Independence Factor')
plt.xticks(range(len(delays)), delays)
plt.yticks(range(len(independence_factors)), independence_factors)

# 3. Box plot of accuracies for different independence factors
plt.subplot(2, 2, 3)
box_data = [results[(i, 3)]['all_accuracies'] for i in independence_factors]
plt.boxplot(box_data, labels=independence_factors)
plt.title('Accuracy Distribution by Independence Factor')
plt.xlabel('Independence Factor')
plt.ylabel('Accuracy')

# 4. Histogram of cascade start positions
plt.subplot(2, 2, 4)
all_cascade_starts = [start for i in independence_factors for d in delays for start in
                      results[(i, d)]['cascade_starts']]
plt.hist(all_cascade_starts, bins=range(1, num_people), align='left', rwidth=0.8)
plt.title('Histogram of Cascade Start Positions')
plt.xlabel('Position in Sequence')
plt.ylabel('Frequency')

plt.tight_layout()

# Save the figure as PNG file
plt.savefig('Restaurant_Improved_Simulation_V1_plot.png')  # Save as PNG file

plt.show()

# Print overall results
for (i, d), res in results.items():
    print(f"Independence Factor: {i}, Delay: {d}")
    print(f"  Average Accuracy: {res['avg_accuracy']:.2f}")
    print(f"  Cascade Probability: {res['cascade_probability']:.2f}")
    print()