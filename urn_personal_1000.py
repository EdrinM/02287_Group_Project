import random
import matplotlib.pyplot as plt
import numpy as np

# Define urns
urn_W = ['white', 'white', 'black']  # UW
urn_B = ['white', 'black', 'black']  # UB

# Simulation parameters
n_agents = 30
n_simulations = 1000
weights = [0.1, 0.3, 0.5, 0.7, 0.9]  # Different weight_personal values
errors = [0.0, 0.1, 0.2, 0.3, 0.4]  # Different error rates

# Function to calculate probabilities with weighting
def calculate_probabilities(draws, prior_UW=0.5, prior_UB=0.5, weight=0.3):
    likelihood_UW = 1
    likelihood_UB = 1
    for draw in draws:
        if draw == 'white':
            likelihood_UW *= 2 / 3
            likelihood_UB *= 1 / 3
        elif draw == 'black':
            likelihood_UW *= 1 / 3
            likelihood_UB *= 2 / 3

    posterior_UW = likelihood_UW * prior_UW
    posterior_UB = likelihood_UB * prior_UB
    normalization = posterior_UW + posterior_UB

    prob_UW = posterior_UW / normalization
    prob_UB = posterior_UB / normalization

    # Apply weight to favor private signals
    weighted_UW = weight * prob_UW + (1 - weight) * prior_UW
    weighted_UB = weight * prob_UB + (1 - weight) * prior_UB

    # Normalize the weighted probabilities
    total = weighted_UW + weighted_UB
    return weighted_UW / total, weighted_UB / total

# Function to simulate an agent's guess
def agent_guess(draws, weight, error_rate):
    prob_UW, prob_UB = calculate_probabilities(draws, weight=weight)
    if random.random() < error_rate:  # Introduce random guessing
        guess = random.choice(['UW', 'UB'])
    else:
        guess = 'UW' if prob_UW >= prob_UB else 'UB'
    confidence = max(prob_UW, prob_UB)
    return guess, confidence


results = {}

# Run simulations for each weight and error rate combination
for weight_personal in weights:
    for error_rate in errors:
        total_accuracy = 0
        cascade_count = 0
        all_accuracies = []
        cascade_starts = []

        for _ in range(n_simulations):
            selected_urn = random.choice([urn_W, urn_B])
            urn_name = 'UW' if selected_urn == urn_W else 'UB'

            draws = []
            blackboard = []
            cascade_start = -1

            # Simulate agents' decisions
            for agent_id in range(1, n_agents + 1):
                draw = random.choice(selected_urn)
                draws.append(draw)

                guess, _ = agent_guess(draws, weight_personal, error_rate)
                blackboard.append(guess)

                # Detect cascade initiation
                if len(blackboard) >= 3:
                    if blackboard[-3:] == ['UW'] * 3 or blackboard[-3:] == ['UB'] * 3:
                        if cascade_start == -1:  # Mark the start of cascade
                            cascade_start = agent_id

            # Verify if the cascade persists for 70% of the remaining sequence
            if cascade_start != -1:
                majority_trend = blackboard[cascade_start - 1]  # Determine cascade decision
                trend_following = sum(1 for g in blackboard[cascade_start:] if g == majority_trend)
                # Validate that cascade dominates the remaining sequence
                if trend_following >= len(blackboard[cascade_start:]) * 0.7:
                    cascade_count += 1  # Count only valid cascades
                    cascade_starts.append(cascade_start)

            # Compute accuracy
            correct_guesses = sum(1 for guess in blackboard if guess == urn_name)
            accuracy = correct_guesses / len(blackboard)
            all_accuracies.append(accuracy)
            total_accuracy += accuracy

        # Store results
        avg_accuracy = total_accuracy / n_simulations
        cascade_probability = cascade_count / n_simulations
        results[(weight_personal, error_rate)] = {
            'avg_accuracy': avg_accuracy,
            'cascade_probability': cascade_probability,
            'all_accuracies': all_accuracies,
            'cascade_starts': cascade_starts,
        }


plt.figure(figsize=(15, 10))

#Heatmap of average accuracy
plt.subplot(2, 2, 1)
accuracy_data = [[results[(w, e)]['avg_accuracy'] for e in errors] for w in weights]
plt.imshow(accuracy_data, cmap='YlOrRd', aspect='auto')
plt.colorbar(label='Average Accuracy')
plt.title('Average Accuracy')
plt.xlabel('Error Rate')
plt.ylabel('Weight Personal')
plt.xticks(range(len(errors)), errors)
plt.yticks(range(len(weights)), weights)

#Heatmap of cascade probability
plt.subplot(2, 2, 2)
cascade_data = [[results[(w, e)]['cascade_probability'] for e in errors] for w in weights]
plt.imshow(cascade_data, cmap='YlOrRd', aspect='auto')
plt.colorbar(label='Cascade Probability')
plt.title('Cascade Probability')
plt.xlabel('Error Rate')
plt.ylabel('Weight Personal')
plt.xticks(range(len(errors)), errors)
plt.yticks(range(len(weights)), weights)

#Box plot of accuracies for weights
plt.subplot(2, 2, 3)
box_data = [results[(w, errors[0])]['all_accuracies'] for w in weights]
plt.boxplot(box_data, labels=weights)
plt.title('Accuracy Distribution by Weight')
plt.xlabel('Weight Personal')
plt.ylabel('Accuracy')

#Histogram of Cascade Start Positions
plt.subplot(2, 2, 4)
all_cascade_starts = [start for w, e in results for start in results[(w, e)]['cascade_starts']]
plt.hist(all_cascade_starts, bins=range(1, n_agents + 1), align='left', rwidth=0.8)
plt.title('Histogram of Cascade Start Positions')
plt.xlabel('Position in Sequence')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

#summary
for (weight, error), data in results.items():
    print(f"Weight: {weight}, Error Rate: {error}")
    print(f"  Avg Accuracy: {data['avg_accuracy']:.2f}")
    print(f"  Cascade Probability: {data['cascade_probability']:.2f}")
    print()
