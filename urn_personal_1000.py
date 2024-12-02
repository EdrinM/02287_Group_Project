import random
import matplotlib.pyplot as plt

# Define urns
urn_W = ['white', 'white', 'black']  # UW: Two white, one black
urn_B = ['white', 'black', 'black']  # UB: One white, two black

# Simulation parameters
n_agents = 30
n_simulations = 1000
weight_personal = 0.3  # Reduce personal weight to prioritize social influence
error_rate = 0.1  # Introduce 10% chance of random guessing

# Track results
cascade_count = 0
accuracy_count = 0
cascade_occurrences = []
confidence_trends = []
final_confidences = []

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

    weighted_UW = posterior_UW * weight
    weighted_UB = posterior_UB * weight
    
    return weighted_UW / normalization, weighted_UB / normalization

# Function to simulate an agent's guess
def agent_guess(agent_id, draw_color, blackboard, draws):
    prob_UW, prob_UB = calculate_probabilities(draws, weight=weight_personal)
    if random.random() < error_rate:  # Introduce random guessing
        guess = random.choice(['UW', 'UB'])
    else:
        guess = 'UW' if prob_UW >= prob_UB else 'UB'
    confidence = max(prob_UW, prob_UB)
    return guess, confidence

# Run simulations
for sim in range(n_simulations):
    selected_urn = random.choice([urn_W, urn_B])
    urn_name = 'UW' if selected_urn == urn_W else 'UB'

    draws = []
    blackboard = []
    confidences = []
    cascade_start = -1

    for agent_id in range(1, n_agents + 1):
        draw = random.choice(selected_urn)
        draws.append(draw)

        guess, confidence = agent_guess(agent_id, draw, blackboard, draws)
        blackboard.append(guess)
        confidences.append(confidence)

        # Detect cascade initiation
        if len(blackboard) >= 3:
            if blackboard[-3:] == ['UW', 'UW', 'UW'] or blackboard[-3:] == ['UB', 'UB', 'UB']:
                if cascade_start == -1:
                    cascade_start = agent_id

    # Store confidence trends
    confidence_trends.append(confidences)

    # Verify cascade
    if cascade_start != -1:
        majority_trend = blackboard[cascade_start - 1]
        trend_following = sum(1 for g in blackboard[cascade_start:] if g == majority_trend)
        if trend_following >= len(blackboard[cascade_start:]) * 0.7:
            cascade_count += 1
            cascade_occurrences.append(1)
        else:
            cascade_occurrences.append(0)
    else:
        cascade_occurrences.append(0)

    # Record final accuracy
    if all(guess == urn_name for guess in blackboard[-3:]):  # Last three guesses
        accuracy_count += 1

    # Record final confidences for visualization
    final_confidences.append(confidences[-1])

# Visualization: Confidence Trends
plt.figure(figsize=(10, 6))
for i, confs in enumerate(confidence_trends[:10]):  # Visualize first 10 runs
    plt.plot(range(1, n_agents + 1), confs, label=f"Simulation {i + 1}")
plt.title("Confidence Trends in First 10 Simulations")
plt.xlabel("Agent Number")
plt.ylabel("Confidence Level")
plt.legend(loc="lower right", fontsize="small", ncol=2)
plt.grid()
plt.show()

# Visualization: Cascade Occurrences
plt.figure(figsize=(8, 5))
plt.bar(['No Cascade', 'Cascade'], [n_simulations - cascade_count, cascade_count], color=['blue', 'orange'])
plt.title("Frequency of Informational Cascades")
plt.ylabel("Number of Simulations")
plt.show()

# Visualization: Accuracy
plt.figure(figsize=(8, 5))
plt.bar(['Incorrect Guesses', 'Correct Guesses'], [n_simulations - accuracy_count, accuracy_count], color=['red', 'green'])
plt.title("Final Accuracy of Simulations")
plt.ylabel("Number of Simulations")
plt.show()

# Visualization: Final Confidence Distribution
plt.figure(figsize=(8, 5))
plt.hist(final_confidences, bins=20, color='purple', alpha=0.7)
plt.title("Distribution of Final Confidence Levels")
plt.xlabel("Confidence Level")
plt.ylabel("Frequency")
plt.show()

# Summary Statistics
print(f"Total Simulations: {n_simulations}")
print(f"Informational Cascades Detected: {cascade_count} ({(cascade_count / n_simulations) * 100:.2f}%)")
print(f"Accuracy of Final Guesses: {accuracy_count} ({(accuracy_count / n_simulations) * 100:.2f}%)")
