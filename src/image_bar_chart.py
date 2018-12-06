# libraries
import numpy as np
import matplotlib.pyplot as plt

bars_0 = [78.83, 78.96, 108.42, 99.02, 66.32, 93.25, 77.23, 92.76, 101.27, 55.67, 94, 62.99, 73.37, 44.62, 77.32, 100.8, 71.21, 56.86, 82.95, 89.91, 69.67, 70.36, 64.7, 84.43, 67.46, 80.33, 70.77, 68.1, 91.9, 83.89, 90.35, 80.3, 78.48]

bars_1 = [62.6, 31.30, 17.3, 59.4, 62.7, 60.2, 57.1, 50.3, 27.3, 46.39, 70.4, 42.1, 41.49, 44.89, 59.09, 45.7, 25.6, 57.4, 28.90, 39.4, 51.5, 45.30, 49.7, 35.30, 66.9, 65.80, 77.4, 57.8, 28.7, 66.0, 32.2, 71.1, 51.89]

bars_2 = [38.6, 39.7, 30.69, 57.7, 34.7, 2.5, -1.69, 30.49, 31.2, 14.9, 23.8, 53.3, 31.0, 22.7, 5.29, 44.00, 26.5, 43.2, 48.0, 67.7, 33.19, 31.8, 53.50, 55.6, 59.2, 46.6, 41.80, 40.5, 12.00, 48.9, 48.3, 25.4, 38.80 ]

bars_3 = [79.89, 62.30, 19.4, 79.0, 81.4, 45.50, 60.90, 48.8, 46.69, 43.1, 58.8, 64.3, 51.8, 51.09, 51.69, 41.59, 43.9, 62.4, 66.5, 45.7, 54.30, 39.6, 52.90, 33.50, 84.5, 69.4, 62.6, 57.90, 56.3, 56.7, 35.59, 71.1, 62.09 ]

bars_4 = [62.6, 45.00, 12.79, 61.79, 72.8, 29.40, 57.1, 83.3, 29.0, 29.20, 67.10, 59.39, 51.8, 57.59, 68.3, 52.8, 37.3, 66.3, 35.30, 52.5, 45.9, 32.4, 47.40, 39.30, 84.5, 65.80, 77.30, 67.6, 26.79, 67.3, 26.79, 71.6, 51.59 ]

bars_5 = [81.2, 54.4, 21.6, 72.1, 81.4, 34.4, 68.7, 48.8, 48.00, 34.90, 70.9, 69.2, 64.19, 62.40, 58.59, 46.39, 31.0, 56.80, 67.4, 45.7, 52.49, 33.0, 40.2, 43.90, 76.6, 44.0, 81.7, 66.0, 57.59, 53.69, 42.29, 63.9, 53.19 ]

bars_6 = [24.0, 56.3, 31.4, 72.10, 30.5, 6.2, 17.29, 27.80, 23.2, 28.6, 46.30, 33.50, 57.99, 20.6, 27.5, 21.59, 31.8, 41.5, 55.5, 96.69, 58.40, 36.4, 31.20, 39.6, 41.90, 49.6, 68.6, 24.3, 48.1, 31.7, 49.60, 53.7, 8.29 ]

bars_7 = [38.5, 31.60, 40.80, 12.69, 15.60, 14.6, 17.4, 1.30, 18.70, 3.79, 21.3, 95.0, 39.0, 26.49, 53.8, 44.59, 13.10, 42.2, 22.69, 50.8, 23.7, 31.20, 34.00, 38.30, 27.3, 52.30, 35.19, 35.5, 14.60, 19.09, 42.2, 37.3, 24.3 ]

# set width of bar
barWidth = 0.1

# Set position of bar on X axis
r0 = np.arange(len(bars_0[0:6]))
r1 = [x + barWidth for x in r0]
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]
r7 = [x + barWidth for x in r6]

# Make the plot
plt.bar(r0, bars_0[0:6], width=barWidth, edgecolor='white', label='Campeão')
plt.bar(r1, bars_1[0:6], width=barWidth, edgecolor='white', label='Redes Neurais')
plt.bar(r2, bars_2[0:6], width=barWidth, edgecolor='white', label='Random Forest')
plt.bar(r3, bars_3[0:6], width=barWidth, edgecolor='white', label='Bayesian Ridge')
plt.bar(r4, bars_4[0:6], width=barWidth, edgecolor='white', label='Ridge')
plt.bar(r5, bars_5[0:6], width=barWidth, edgecolor='white', label='ElasticNet')
plt.bar(r6, bars_6[0:6], width=barWidth, edgecolor='white', label='Gradient Boosting')
plt.bar(r7, bars_7[0:6], width=barWidth, edgecolor='white', label='SVR')

# Add xticks on the middle of the group bars
plt.xlabel('Rodada')
plt.ylabel('Pontos')

plt.xticks([r + barWidth for r in range(len(bars_0[0:6]))], ['6', '7', '8', '9', '10', '11'])

# Create legend & Show graphic
plt.legend(loc='best')
plt.show()

# Make the plot
plt.bar(r0, bars_0[6:12], width=barWidth, edgecolor='white', label='Campeão')
plt.bar(r1, bars_1[6:12], width=barWidth, edgecolor='white', label='Redes Neurais')
plt.bar(r2, bars_2[6:12], width=barWidth, edgecolor='white', label='Random Forest')
plt.bar(r3, bars_3[6:12], width=barWidth, edgecolor='white', label='Bayesian Ridge')
plt.bar(r4, bars_4[6:12], width=barWidth, edgecolor='white', label='Ridge')
plt.bar(r5, bars_5[6:12], width=barWidth, edgecolor='white', label='ElasticNet')
plt.bar(r6, bars_6[6:12], width=barWidth, edgecolor='white', label='Gradient Boosting')
plt.bar(r7, bars_7[6:12], width=barWidth, edgecolor='white', label='SVR')

# Add xticks on the middle of the group bars
plt.xlabel('Rodada')
plt.ylabel('Pontos')

plt.xticks([r + barWidth for r in range(len(bars_0[6:12]))], ['12', '13', '14', '15', '16', '17'])

# Create legend & Show graphic
plt.show()

# Make the plot
plt.bar(r0, bars_0[12:18], width=barWidth, edgecolor='white', label='Campeão')
plt.bar(r1, bars_1[12:18], width=barWidth, edgecolor='white', label='Redes Neurais')
plt.bar(r2, bars_2[12:18], width=barWidth, edgecolor='white', label='Random Forest')
plt.bar(r3, bars_3[12:18], width=barWidth, edgecolor='white', label='Bayesian Ridge')
plt.bar(r4, bars_4[12:18], width=barWidth, edgecolor='white', label='Ridge')
plt.bar(r5, bars_5[12:18], width=barWidth, edgecolor='white', label='ElasticNet')
plt.bar(r6, bars_6[12:18], width=barWidth, edgecolor='white', label='Gradient Boosting')
plt.bar(r7, bars_7[12:18], width=barWidth, edgecolor='white', label='SVR')

# Add xticks on the middle of the group bars
plt.xlabel('Rodada')
plt.ylabel('Pontos')

plt.xticks([r + barWidth for r in range(len(bars_0[12:18]))], ['18', '19', '20', '21', '22', '23'])

# Create legend & Show graphic
plt.show()

# Make the plot
plt.bar(r0, bars_0[18:24], width=barWidth, edgecolor='white', label='Campeão')
plt.bar(r1, bars_1[18:24], width=barWidth, edgecolor='white', label='Redes Neurais')
plt.bar(r2, bars_2[18:24], width=barWidth, edgecolor='white', label='Random Forest')
plt.bar(r3, bars_3[18:24], width=barWidth, edgecolor='white', label='Bayesian Ridge')
plt.bar(r4, bars_4[18:24], width=barWidth, edgecolor='white', label='Ridge')
plt.bar(r5, bars_5[18:24], width=barWidth, edgecolor='white', label='ElasticNet')
plt.bar(r6, bars_6[18:24], width=barWidth, edgecolor='white', label='Gradient Boosting')
plt.bar(r7, bars_7[18:24], width=barWidth, edgecolor='white', label='SVR')

# Add xticks on the middle of the group bars
plt.xlabel('Rodada')
plt.ylabel('Pontos')

plt.xticks([r + barWidth for r in range(len(bars_0[18:24]))], ['24', '25', '26', '27', '28', '29'])

# Create legend & Show graphic
plt.show()

# Make the plot
plt.bar(r0, bars_0[24:30], width=barWidth, edgecolor='white', label='Campeão')
plt.bar(r1, bars_1[24:30], width=barWidth, edgecolor='white', label='Redes Neurais')
plt.bar(r2, bars_2[24:30], width=barWidth, edgecolor='white', label='Random Forest')
plt.bar(r3, bars_3[24:30], width=barWidth, edgecolor='white', label='Bayesian Ridge')
plt.bar(r4, bars_4[24:30], width=barWidth, edgecolor='white', label='Ridge')
plt.bar(r5, bars_5[24:30], width=barWidth, edgecolor='white', label='ElasticNet')
plt.bar(r6, bars_6[24:30], width=barWidth, edgecolor='white', label='Gradient Boosting')
plt.bar(r7, bars_7[24:30], width=barWidth, edgecolor='white', label='SVR')

# Add xticks on the middle of the group bars
plt.xlabel('Rodada')
plt.ylabel('Pontos')

plt.xticks([r + barWidth for r in range(len(bars_0[24:30]))], ['30', '31', '32', '33', '34', '35'])

# Create legend & Show graphic
plt.show()

# Set position of bar on X axis
r0 = np.arange(len(bars_0[30:33]))
r1 = [x + barWidth for x in r0]
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]
r7 = [x + barWidth for x in r6]

# Make the plot
plt.bar(r0, bars_0[30:33], width=barWidth, edgecolor='white', label='Campeão')
plt.bar(r1, bars_1[30:33], width=barWidth, edgecolor='white', label='Redes Neurais')
plt.bar(r2, bars_2[30:33], width=barWidth, edgecolor='white', label='Random Forest')
plt.bar(r3, bars_3[30:33], width=barWidth, edgecolor='white', label='Bayesian Ridge')
plt.bar(r4, bars_4[30:33], width=barWidth, edgecolor='white', label='Ridge')
plt.bar(r5, bars_5[30:33], width=barWidth, edgecolor='white', label='ElasticNet')
plt.bar(r6, bars_6[30:33], width=barWidth, edgecolor='white', label='Gradient Boosting')
plt.bar(r7, bars_7[30:33], width=barWidth, edgecolor='white', label='SVR')

# Add xticks on the middle of the group bars
plt.xlabel('Rodada')
plt.ylabel('Pontos')

plt.xticks([r + barWidth for r in range(len(bars_0[30:33]))], ['36', '37', '38'])

# Create legend & Show graphic
plt.show()
