from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive Agg backend
import matplotlib.pyplot as plt

from scipy.integrate import odeint
import io
import base64

app = Flask(__name__)

# SIR model implementation
def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Function to generate SIR plots
def generate_sir_plot(beta, gamma):
    N = 1000  # Total population
    I0 = 1    # Initial number of infectious individuals
    R0 = 0    # Initial number of recovered individuals
    S0 = N - I0 - R0  # Initial number of susceptible individuals

    t = np.linspace(0, 160, 160)  # Time points (in days)
    y0 = [S0 / N, I0 / N, R0 / N]  # Normalize population

    solution = odeint(sir_model, y0, t, args=(beta, gamma))
    S, I, R = solution.T

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptible', color='blue')
    plt.plot(t, I, label='Infectious', color='red')
    plt.plot(t, R, label='Recovered', color='green')
    plt.title(f'SIR Model Simulation (beta={beta}, gamma={gamma})')
    plt.xlabel('Time (days)')
    plt.ylabel('Proportion of Population')
    plt.legend()
    plt.grid(True)

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return plot_url

@app.route('/')
def index():
    return render_template('index.html')

# Function to generate SIR plots with observed and predicted data
def generate_sir_plot(beta, gamma, observed_days=50, predicted_days=160):
    N = 1000  # Total population
    I0 = 1    # Initial number of infectious individuals
    R0 = 0    # Initial number of recovered individuals
    S0 = N - I0 - R0  # Initial number of susceptible individuals

    t = np.linspace(0, predicted_days, predicted_days)  # Time points (in days)
    y0 = [S0 / N, I0 / N, R0 / N]  # Normalize population

    # Generate observed data
    observed_t = t[:observed_days]
    observed_solution = odeint(sir_model, y0, observed_t, args=(beta, gamma))

    # Generate full prediction
    full_solution = odeint(sir_model, y0, t, args=(beta, gamma))

    S_obs, I_obs, R_obs = observed_solution.T
    S_pred, I_pred, R_pred = full_solution.T

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(observed_t, I_obs, label='Observed Infectious', color='orange')
    plt.plot(t, I_pred, label='Predicted Infectious', color='red', linestyle='dashed')
    plt.title(f'SIR Model Prediction (beta={beta}, gamma={gamma})')
    plt.xlabel('Time (days)')
    plt.ylabel('Proportion of Infectious Population')
    plt.legend()
    plt.grid(True)

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return plot_url

@app.route('/update', methods=['POST'])
def update_plot():
    data = request.get_json()
    beta = float(data['beta'])
    gamma = float(data['gamma'])
    observed_days = int(data.get('observed_days', 50))  # Default to 50 days
    predicted_days = int(data.get('predicted_days', 160))  # Default to 160 days

    plot_url = generate_sir_plot(beta, gamma, observed_days, predicted_days)

    return jsonify({'plot_url': plot_url})

# @app.route('/update', methods=['POST'])
# def update_plot():
#     data = request.get_json()
#     beta = float(data['beta'])
#     gamma = float(data['gamma'])
#
#     plot_url = generate_sir_plot(beta, gamma)
#
#     return jsonify({'plot_url': plot_url})

if __name__ == '__main__':
    app.run(debug=True,port=8000)