import numpy as np
import matplotlib.pyplot as plt

def ibm_model(no_person, timestep, recovery_rate, new_births, transmission_rate, death_rate, final_timestep,
              num_simulations, initial_S, initial_I, initial_R):
    def initialize_population(no_person, death_rate, initial_S, initial_I, initial_R):
        states = np.zeros(no_person)
        states[:initial_S] = 0  # Susceptible
        states[initial_S:initial_S + initial_I] = 1  # Infected
        states[initial_S + initial_I:] = 2  # Recovered

        ages = np.random.uniform(0, 100, size=no_person)
        age_death = np.random.exponential(scale=1 / death_rate, size=no_person)
        days_infected = -1 * np.ones(no_person)
        days_to_recovery = np.random.exponential(scale=1 / recovery_rate, size=no_person)
        return np.column_stack((states, ages, age_death, days_infected, days_to_recovery))

    all_susceptible_population = []
    all_infected_population = []
    all_recovered_population = []

    plt.figure(figsize=(10, 6))

    for sim in range(num_simulations):
        P0 = initialize_population(no_person, death_rate, initial_S, initial_I, initial_R)

        susceptible_population = [initial_S]
        infected_population = [initial_I]
        recovered_population = [initial_R]
        print(f"Timestep 1: Susceptible={initial_S}, Infected={initial_I}, Recovered={initial_R}")

        for t in range(1, final_timestep + 1):
            if t == 1:
                # For the first timestep, set the counts to the specific initial values
                susceptible_population.append(initial_S)
                infected_population.append(initial_I)
                recovered_population.append(initial_R)
                continue

            # Death of Individuals
            age_condition = P0[:, 1] > P0[:, 2]
            P0 = np.delete(P0, np.where(age_condition)[0], axis=0)

            # Birth of New Individuals
            bth = np.random.poisson(new_births * timestep * no_person)
            new_individuals = initialize_population(bth, death_rate, initial_S, initial_I, initial_R)
            P0 = np.vstack((P0, new_individuals))

            # Ensure the population size remains constant
            current_population = len(P0)
            if current_population < no_person:
                additional_born = no_person - current_population
                extra_new_individuals = initialize_population(additional_born, death_rate, initial_S, initial_I,
                                                              initial_R)
                P0 = np.vstack((P0, extra_new_individuals))

            # Recovery of Infected Individuals
            indr = np.where((P0[:, 0] == 1) & (P0[:, 3] >= P0[:, 4]))[0]
            P0[indr, 0] = 2  # Set recovered
            P0[indr, 3:5] = 0

            # Infection Spread
            for n in range(len(P0)):
                if P0[n, 0] == 0:  # Susceptible individual
                    if np.random.rand() < transmission_rate:
                        P0[n, 0] = 1  # Infection
                        P0[n, 3] = 0
                        P0[n, 4] = np.random.exponential(scale=1 / recovery_rate)

            # Update Time and Recovery Status
            P0[:, 3] += timestep  # Increment days_infected for infected individuals
            P0[:, 4] = np.maximum(P0[:, 4] - timestep, 0)

            # Set individuals to recovered
            indr = np.where((P0[:, 0] == 1) & (P0[:, 3] >= P0[:, 4]))[0]
            P0[indr, 0] = 2  # Set recovered
            P0[indr, 3:5] = 0

            # Check and adjust population limit
            total_population = np.sum(P0[:, 0] == 0) + np.sum(P0[:, 0] == 1) + np.sum(P0[:, 0] == 2)
            scaling_factor = no_person / total_population
            new_total_population = int(total_population * scaling_factor)
            if new_total_population > no_person:
                # Adjust populations to maintain the limit
                diff = new_total_population - no_person
                P0[np.random.choice(P0.shape[0], diff, replace=False), 0] = -1

            # Calculate and store population in each compartment
            # Calculate and store population in each compartment
            new_susceptible = int(np.sum(P0[:, 0] == 0) * scaling_factor)
            if new_susceptible < susceptible_population[-1]:
                susceptible_population.append(new_susceptible)
            elif susceptible_population[-1] > 0:
                susceptible_population.append(susceptible_population[-1] - 1)
            else:
                susceptible_population.append(0)


            # Calculate infected individuals
            infected = int(np.sum(P0[:, 0] == 1)* transmission_rate * scaling_factor)

            # Calculate recovered individuals based on the constraints
            recovered = int(no_person - new_susceptible - infected)
            if recovered < 0:
                # Adjust recovered to ensure it's non-negative
                recovered = 0

            if recovered + infected > no_person:
                # If the sum of recovered and infected exceeds the total population, adjust infected
                infected = no_person - recovered




            # Update infected and recovered populations
            infected_population.append(infected)

            recovered_population.append(recovered)
            # Check and adjust total population to ensure it equals no_person
            total_population_check = susceptible_population[-1] + infected_population[-1] + recovered_population[-1]
            if total_population_check != no_person:
                diff = no_person - total_population_check
                if recovered_population[-1] + diff >= 0:
                    recovered_population[-1] += diff
                else:
                    recovered_population[-1] = 0

            print(
                f"Timestep {t}: Total Population={new_total_population}, Susceptible={susceptible_population[-1]}, Infected={infected_population[-1]}, Recovered={recovered_population[-1]}")

        all_susceptible_population.append(susceptible_population)
        all_infected_population.append(infected_population)
        all_recovered_population.append(recovered_population)

        plt.plot(range(final_timestep + 1), susceptible_population, 'b-', alpha=0.2)
        plt.plot(range(final_timestep + 1), infected_population, 'r-', alpha=0.2)
        plt.plot(range(final_timestep + 1), recovered_population, 'g-', alpha=0.2)

    avg_susceptible = np.mean(all_susceptible_population, axis=0)
    avg_infected = np.mean(all_infected_population, axis=0)
    avg_recovered = np.mean(all_recovered_population, axis=0)

    plt.plot(range(final_timestep + 1), avg_susceptible, label='Average Susceptible', color='black', linestyle='--')
    plt.plot(range(final_timestep + 1), avg_infected, label='Average Infected', color='yellow', linestyle='--')
    plt.plot(range(final_timestep + 1), avg_recovered, label='Average Recovered', color='brown', linestyle='--')

    plt.xlabel('Time Steps (Days)')
    plt.ylabel('Population')
    plt.title('IBM Model with Population Limit of 1000')
    plt.legend()
    plt.grid(True)
    plt.show()


# Parameters
no_person = 1000
timestep = 0.1
recovery_rate = 1/3
new_births = 1/50
transmission_rate = 0.25
death_rate = 1/60
final_timestep = 100
num_simulations = 100

# Initial values for S, I, R
initial_S = 900
initial_I = 90
initial_R = 10

# Run the IBM model with multiple simulations
ibm_model(no_person, timestep, recovery_rate, new_births, transmission_rate, death_rate, final_timestep,
          num_simulations, initial_S, initial_I, initial_R)
