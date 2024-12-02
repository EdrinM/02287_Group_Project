import random
import matplotlib.pyplot as plt

# Define urns
urn_W = ['white', 'white', 'black']  # UW: Two white, one black
urn_B = ['white', 'black', 'black']  # UB: One white, two black

# Randomly select an urn
selected_urn = random.choice([urn_W, urn_B])
urn_name = 'UW' if selected_urn == urn_W else 'UB'

# Number of agents
n_agents = 10

# Blackboard to track guesses and confidences
blackboard = []

# Function to calculate probabilities with weighting
def calculate_probabilities(draws, prior_UW=0.5, prior_UB=0.5, weight=0.7):
    # Likelihoods
    likelihood_UW = 1
    likelihood_UB = 1
    for draw in draws:
        if draw == 'white':
            likelihood_UW *= 2 / 3  # P(white|UW)
            likelihood_UB *= 1 / 3  # P(white|UB)
        elif draw == 'black':
            likelihood_UW *= 1 / 3  # P(black|UW)
            likelihood_UB *= 2 / 3  # P(black|UB)
    
    # Posteriors
    posterior_UW = likelihood_UW * prior_UW
    posterior_UB = likelihood_UB * prior_UB
    normalization = posterior_UW + posterior_UB
    
    # Apply weighting to emphasize the agent's draw
    weighted_UW = posterior_UW * weight
    weighted_UB = posterior_UB * weight
    
    return weighted_UW / normalization, weighted_UB / normalization

# Function to simulate an agent's guess
def agent_guess(agent_id, draw_color, blackboard, draws, weight_personal=0.7):
    # Calculate probabilities based on blackboard and personal draw
    prob_UW, prob_UB = calculate_probabilities(draws, weight=weight_personal)
    
    # Guess the urn with higher probability
    guess = 'UW' if prob_UW >= prob_UB else 'UB'
    confidence = max(prob_UW, prob_UB)
    return guess, confidence

# Simulate agents taking turns
draws = []  # Record of all ball draws
confidence_over_time = []  # Track confidence levels for visualization
for agent_id in range(1, n_agents + 1):
    # Each agent draws a ball from the urn
    draw = random.choice(selected_urn)
    draws.append(draw)

    # Agent makes a guess
    guess, confidence = agent_guess(agent_id, draw, blackboard, draws, weight_personal=0.7)
    blackboard.append((guess, confidence))
    confidence_over_time.append((agent_id, confidence))

    # Print the results for this agent
    print(f"Agent {agent_id} drew {draw}, guessed {guess} with confidence {confidence:.2f}")

# Final results
print("\nSimulation complete.")
print(f"Actual urn: {urn_name}")
print(f"Blackboard guesses: {[guess for guess, _ in blackboard]}")

# Visualization
agents, confidences = zip(*confidence_over_time)
plt.figure(figsize=(10, 6))
plt.plot(agents, confidences, marker='o', label='Confidence')
plt.axhline(y=1.0, color='green', linestyle='--', label='Perfect Confidence')
plt.axhline(y=0.5, color='red', linestyle='--', label='Initial Uncertainty')
plt.title('Agent Confidence Levels Over Time (Reduced Cascades)')
plt.xlabel('Agent Number')
plt.ylabel('Confidence in Guess')
plt.legend()
plt.grid()
plt.show()
