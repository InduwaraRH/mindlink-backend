# ALGORITHM JITAI_Online_Learning
#     INPUT: 
#         Context Vector x_t (Normalized: Time, Mood, Stress)
#         Reward Signal r_t (1.0 if clicked, 0.0 if ignored)
#         Current Weights w_t
#         Learning Rate η (eta) = 0.01

#     OUTPUT: 
#         Updated Weights w_{t+1}

#     BEGIN
#         // Step 1: Prediction (Inference)
#         Calculate score = DotProduct(w_t, x_t)
#         Probability p = Sigmoid(score)
        
#         // Step 2: Loss Calculation (Log-Loss)
#         // We want to minimize the difference between Prediction (p) and Reward (r_t)
#         Error e = p - r_t
        
#         // Step 3: Gradient Descent Update (The Learning Step)
#         // Adjust weights in the opposite direction of the error
#         Gradient ∇ = e * x_t
        
#         w_{t+1} ← w_t - (η * ∇)

#         // Step 4: Regularization (Elastic Net)
#         // Apply L2 penalty to prevent overfitting
#         w_{t+1} ← ApplyPenalty(w_{t+1})

#         RETURN w_{t+1}
#     END