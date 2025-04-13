import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs, load_iris, fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler

print("1. Собствени стойности и собствени вектори")

# Проста матрица
A = np.array([[2, -4], [-1, -1]])
print(f"Матрица A:\n{A}")

# Собствени стойности и вектори
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Собствени стойности: {eigenvalues}")
print(f"Собствени вектори:\n{eigenvectors}")

# A*v = λ*v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]  # i-ти собствен вектор
    Av = A.dot(v)           # A*v
    lambdav = eigenvalues[i] * v  # λ*v
    print(f"\nПроверка за собствен вектор {i+1}:")
    print(f"A*v{i+1} = {Av}")
    print(f"λ{i+1}*v{i+1} = {lambdav}")
    print(f"Разлика: {np.abs(Av - lambdav).sum():.10f}")  # Трябва да е близко до 0

print("\n2. Собствен базис и спектър на матрица")
print(f"Спектър на матрицата A: {eigenvalues}")
print(f"Собствен базис на матрицата A:\n{eigenvectors}")

det_eigenvectors = np.linalg.det(eigenvectors)
print(f"Детерминанта на матрицата от собствени вектори: {det_eigenvectors}")
print("Собствените вектори са линейно независими, когато детерминантата е различна от 0.")

print("\n3. Собствени стойности и вектори")
print("Изчисляване на характеристичното уравнение на матрица")

I = np.eye(2)  # Единична матрица 2x2

# Аналитично характеристично уравнение
print("Характеристично уравнение det(A - λI) = 0:")
print("det([ [2-λ, -4], [-1, -1-λ] ]) = 0")
print("(2-λ)(-1-λ) - (-4)(-1) = 0")
print("(2-λ)(-1-λ) - 4 = 0")
print("λ² - λ + 2λ + 4 = 0")
print("λ² + λ + 4 = 0")

# Решаване на квадратното уравнение
a, b, c = 1, 1, 4
disc = b**2 - 4*a*c
print(f"Дискриминанта: {disc}")
if disc < 0:
    real_part = -b / (2*a)
    imag_part = np.sqrt(abs(disc)) / (2*a)
    roots = [complex(real_part, imag_part), complex(real_part, -imag_part)]
    print(f"Комплексни корени: {roots}")
else:
    roots = [(-b + np.sqrt(disc)) / (2*a), (-b - np.sqrt(disc)) / (2*a)]
    print(f"Реални корени: {roots}")

print(f"Сравнение с numpy: {eigenvalues}")

print("\n4. Визуализация на проекция")

v = np.array([3, 4])  # Вектор
u1 = np.array([1, 0])  # единичен вектор по x
u2 = np.array([0, 1])  # единичен вектор по y

proj_x = np.dot(v, u1) * u1  # Проекция върху x-ос
proj_y = np.dot(v, u2) * u2  # Проекция върху y-ос

plt.figure(figsize=(8, 8))
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='blue', label='v = [3, 4]')
plt.quiver(0, 0, proj_x[0], proj_x[1], angles='xy', scale_units='xy', scale=1, color='red', label='proj_x(v)')
plt.quiver(0, 0, proj_y[0], proj_y[1], angles='xy', scale_units='xy', scale=1, color='green', label='proj_y(v)')

plt.plot([v[0], v[0]], [0, v[1]], 'k--', alpha=0.3)
plt.plot([0, v[0]], [v[1], v[1]], 'k--', alpha=0.3)

plt.grid(True)
plt.axis('equal')
plt.xlim(-1, 5)
plt.ylim(-1, 5)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.legend()
plt.title('Проекция на вектор върху координатните оси')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

print("\n5. Как проекцията запазва форми")

# Създаваме точки, които формират окръжност в 3D
theta = np.linspace(0, 2*np.pi, 100)
r = 1
x = r * np.cos(theta)
y = r * np.sin(theta)
z = np.zeros_like(theta)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, 'b-', label='Окръжност в 3D')

ax.plot(x, y, np.zeros_like(x), 'r--', label='Проекция върху xy-равнина')
ax.plot(x, np.zeros_like(y), z, 'g--', label='Проекция върху xz-равнина')
ax.plot(np.zeros_like(x), y, z, 'm--', label='Проекция върху yz-равнина')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Окръжност в 3D и нейните проекции')
ax.legend()
plt.show()

# Проекции при различни ъгли - визуализация как се запазват форми
square_x = np.array([0, 1, 1, 0, 0])
square_y = np.array([0, 0, 1, 1, 0])
square_z = np.zeros_like(square_x)

# Създаваме трансформации за различни ъгли на гледане
angles = [0, 30, 60, 85]
fig, axs = plt.subplots(1, len(angles), figsize=(15, 4))

for i, angle in enumerate(angles):
    # Създаваме матрица за завъртане
    theta = np.radians(angle)
    rotation = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # Завъртане
    rotated_points = np.dot(rotation, np.vstack([square_x, square_y]))
    
    axs[i].plot(rotated_points[0], rotated_points[1], 'b-')
    axs[i].set_aspect('equal')
    axs[i].set_xlim(-1.5, 1.5)
    axs[i].set_ylim(-1.5, 1.5)
    axs[i].grid(True)
    axs[i].set_title(f'Ъгъл: {angle}°')

plt.suptitle('Проекции на квадрат при различни ъгли на гледане')
plt.tight_layout()
plt.show()

print("\n6. Връзка между проекцията и собствените стойности/вектори")

np.random.seed(42)
mean = [0, 0]
cov = [[3, 2], [2, 2]]  # Ковариационна матрица - определя формата и ориентацията
data = np.random.multivariate_normal(mean, cov, 500)

cov_matrix = np.cov(data, rowvar=False)
print(f"Ковариационна матрица на данните:\n{cov_matrix}")

eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"Собствени стойности на ковариационната матрица: {eigenvalues}")
print(f"Собствени вектори на ковариационната матрица:\n{eigenvectors}")

plt.figure(figsize=(10, 8))
plt.scatter(data[:, 0], data[:, 1], alpha=0.5)

# Собствени вектори, мащабирани с квадратен корен от собствените стойности
for i in range(2):
    plt.quiver(mean[0], mean[1], 
              eigenvectors[0, i] * np.sqrt(eigenvalues[i]), 
              eigenvectors[1, i] * np.sqrt(eigenvalues[i]),
              angles='xy', scale_units='xy', scale=1, 
              color=['red', 'green'][i],
              label=f'Собствен вектор {i+1}, λ={eigenvalues[i]:.2f}')

plt.axis('equal')
plt.grid(True)
plt.legend()
plt.title('Данни и собствени вектори на ковариационната матрица')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

print("\n7. PCA - имплементация от нулата")

def my_pca(X, n_components):
    """
    Параметри:
    X : numpy array, форма (n_samples, n_features)
        Входящите данни
    n_components : int
        Брой главни компоненти, които да се върнат
        
    Връща:
    X_pca : numpy array, форма (n_samples, n_components)
        Проектираните данни
    components : numpy array, форма (n_components, n_features)
        Главните компоненти (собствените вектори)
    explained_variance : numpy array, форма (n_components,)
        Обяснената вариация за всеки компонент
    """
    # Стъпка 1: Центриране на данните към средната)
    X_centered = X - np.mean(X, axis=0)
    
    # Стъпка 2: Изчисляване на ковариационната матрица
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # Стъпка 3: Изчисляване на собствените стойности и вектори
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Стъпка 4: Сортиране на собствените стойности и вектори в низходящ ред
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Стъпка 5: Избиране на top n_components
    components = eigenvectors[:, :n_components]
    explained_variance = eigenvalues[:n_components]
    
    # Стъпка 6: Проектиране на данните върху главните компоненти
    X_pca = X_centered.dot(components)
    
    return X_pca, components, explained_variance

# ТЕстване на имплементацията върху генерирани данни
np.random.seed(42)
# Данни с 3 характеристики
X = np.random.rand(100, 3) * 10
print(f"Форма на оригиналните данни: {X.shape}")

# Прилагане на PCA имплементация
X_pca, components, explained_variance = my_pca(X, n_components=2)
print(f"Форма на редуцираните данни: {X_pca.shape}")
print(f"Форма на компонентите: {components.shape}")
print(f"Обяснена вариация: {explained_variance}")

# Общата вариация и процента обяснена вариация
total_variance = np.sum(np.var(X - np.mean(X, axis=0), axis=0))
explained_variance_ratio = explained_variance / total_variance
print(f"Процент обяснена вариация: {explained_variance_ratio * 100}")
print(f"Общо обяснена вариация: {np.sum(explained_variance_ratio) * 100:.2f}%")

print("\n8. Главните компоненти")
print(f"Брой оригинални характеристики в данните: {X.shape[1]}")
print(f"Максимален брой главни компоненти: {X.shape[1]}")
print(f"Избран брой главни компоненти: 2")

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
plt.grid(True)
plt.title('Проекция на данните върху първите 2 главни компонента')
plt.xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)')
plt.show()

print("\n9. Дисперсия и обяснена дисперсия")

# Дисперсията по всяка оригинална характеристика
original_variance = np.var(X, axis=0)
print(f"Дисперсия по оригиналните характеристики: {original_variance}")
print(f"Обща дисперсия: {np.sum(original_variance)}")

# Дисперсията по всеки главен компонент
pca_variance = np.var(X_pca, axis=0)
print(f"Дисперсия по главните компоненти: {pca_variance}")
print(f"Обща дисперсия след PCA: {np.sum(pca_variance)}")

print("\n10. Връзка между главните компоненти и обяснената дисперсия")

# Изчисляване на PCA за всички възможни компоненти
X_full_pca, full_components, full_explained_variance = my_pca(X, n_components=X.shape[1])

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(full_explained_variance) + 1), full_explained_variance, alpha=0.7)
plt.grid(True, axis='y')
plt.xlabel('Главен компонент')
plt.ylabel('Обяснена дисперсия')
plt.title('Обяснена дисперсия за всеки главен компонент')
plt.xticks(range(1, len(full_explained_variance) + 1))
plt.show()

cumulative_variance_ratio = np.cumsum(full_explained_variance) / np.sum(full_explained_variance)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'o-', markersize=8)
plt.grid(True)
plt.xlabel('Брой компоненти')
plt.ylabel('Кумулативна обяснена дисперсия')
plt.title('Кумулативна обяснена дисперсия спрямо броя компоненти')
plt.xticks(range(1, len(cumulative_variance_ratio) + 1))

# Линии на праговите стойности
for threshold in [0.7, 0.8, 0.9, 0.95]:
    components_needed = np.argmax(cumulative_variance_ratio >= threshold) + 1
    plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=components_needed, color='r', linestyle='--', alpha=0.3)
    plt.text(components_needed + 0.1, threshold - 0.03, f'{components_needed} комп.: {threshold*100:.0f}%', 
             fontsize=9, color='red')

plt.ylim([0, 1.05])
plt.show()

# PCA - Iris dataset
print("\n11. Демонстрация на PCA върху реални данни - Iris dataset")

# Зареждане на Iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"Форма на Iris dataset: {X_iris.shape}")
print(f"Характеристики: {feature_names}")
print(f"Класове: {target_names}")

# Стандартизиране на данните
scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)

# Прилагане на PCA
X_iris_pca, iris_components, iris_explained_variance = my_pca(X_iris_scaled, n_components=2)

# Изчисляване на дисперсия
total_variance = np.sum(np.var(X_iris_scaled, axis=0))
explained_variance_ratio = iris_explained_variance / total_variance

print(f"Обяснена дисперсия за всеки компонент: {iris_explained_variance}")
print(f"Процент обяснена дисперсия: {explained_variance_ratio * 100}")
print(f"Общо обяснена дисперсия: {np.sum(explained_variance_ratio) * 100:.2f}%")

# Ррезултати от PCA
plt.figure(figsize=(10, 8))
colors = ['navy', 'turquoise', 'darkorange']

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_iris_pca[y_iris == i, 0], X_iris_pca[y_iris == i, 1], 
               color=color, alpha=0.8, lw=2, label=target_name)
    
plt.title('PCA на Iris dataset')
plt.xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Визуализация на главните компоненти
plt.figure(figsize=(12, 6))

# Създаване на heatmap за компонентите
plt.imshow(iris_components.T, cmap='viridis', aspect='auto')
plt.yticks(range(len(feature_names)), feature_names)
plt.xticks([0, 1], [f'PC1 ({explained_variance_ratio[0]*100:.1f}%)',
                    f'PC2 ({explained_variance_ratio[1]*100:.1f}%)'])
plt.colorbar(label='Тегло на компонента')
plt.title('Тегла на главните компоненти за Iris dataset')
plt.tight_layout()
plt.show()

# Приложения на PCA: редуциране на 3D данни до 2D
print("\n12. PCA: редуциране на 3D данни до 2D")

# Генериране на 3D данни с клъстери
np.random.seed(42)
X_3d, y_3d = make_blobs(n_samples=300, centers=4, n_features=3, random_state=42)

# Прилагане на PCA
X_3d_pca, components_3d, explained_variance_3d = my_pca(X_3d, n_components=2)

# Изчисляване на дисперсията в проценти
total_variance_3d = np.sum(np.var(X_3d, axis=0))
explained_variance_ratio_3d = explained_variance_3d / total_variance_3d

print(f"Обяснена дисперсия за всеки компонент: {explained_variance_3d}")
print(f"Процент обяснена дисперсия: {explained_variance_ratio_3d * 100}")
print(f"Общо обяснена дисперсия: {np.sum(explained_variance_ratio_3d) * 100:.2f}%")

# Визуализация на 3D данни и 2D проекция
fig = plt.figure(figsize=(12, 5))

# Оригинални 3D данни
ax1 = fig.add_subplot(121, projection='3d')
scatter = ax1.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y_3d, cmap='viridis', s=30)
ax1.set_title('Оригинални 3D данни')
ax1.set_xlabel('Характеристика 1')
ax1.set_ylabel('Характеристика 2')
ax1.set_zlabel('Характеристика 3')

# 2D проекция след PCA
ax2 = fig.add_subplot(122)
scatter = ax2.scatter(X_3d_pca[:, 0], X_3d_pca[:, 1], c=y_3d, cmap='viridis', s=30)
ax2.set_title('2D проекция след PCA')
ax2.set_xlabel(f'PC1 ({explained_variance_ratio_3d[0]*100:.1f}%)')
ax2.set_ylabel(f'PC2 ({explained_variance_ratio_3d[1]*100:.1f}%)')
ax2.grid(True)

plt.tight_layout()
plt.show()

print("\n13. Практическо приложение: визуализация на данни с висока размерност")

#Симулация на данни с висока размерност (15D) и визуализиране в 3D
np.random.seed(42)

# Случайна матрица за генериране на корелирани данни
random_mat = np.random.randn(15, 15)
transformation = random_mat.dot(random_mat.T)  # положително полуопределена матрица

# Генериране на 15D данни с 5 класа
n_samples = 500
n_classes = 5

# Генериране на клъстери с различни центрове
X_high_dim = np.zeros((n_samples, 15))
y_high_dim = np.zeros(n_samples, dtype=int)

samples_per_class = n_samples // n_classes
centers = np.random.randn(n_classes, 15) * 5  # различни центрове за всеки клас

for i in range(n_classes):
    start_idx = i * samples_per_class
    end_idx = (i + 1) * samples_per_class
    
    # Генериране на данни около центъра за този клас
    X_high_dim[start_idx:end_idx] = np.random.multivariate_normal(
        centers[i], transformation, samples_per_class)
    
    y_high_dim[start_idx:end_idx] = i

print(f"Форма на генерираните 15D данни: {X_high_dim.shape}")

# Прилагане на PCA
X_high_dim_pca, components_high_dim, explained_variance_high_dim = my_pca(X_high_dim, n_components=3)

# Изчисляване на дисперсията в проценти
total_variance_high_dim = np.sum(np.var(X_high_dim, axis=0))
explained_variance_ratio_high_dim = explained_variance_high_dim / total_variance_high_dim

print(f"Обяснена дисперсия за всеки компонент: {explained_variance_high_dim[:3]}")
print(f"Процент обяснена дисперсия: {explained_variance_ratio_high_dim * 100}")
print(f"Общо обяснена дисперсия: {np.sum(explained_variance_ratio_high_dim) * 100:.2f}%")

# Визуализация на 3D проекция от 15D данни
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    X_high_dim_pca[:, 0], X_high_dim_pca[:, 1], X_high_dim_pca[:, 2],
    c=y_high_dim, 
    cmap='tab10', 
    s=50,
    alpha=0.7
)

ax.set_title('3D PCA проекция на 15D данни')
ax.set_xlabel(f'PC1 ({explained_variance_ratio_high_dim[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({explained_variance_ratio_high_dim[1]*100:.1f}%)')
ax.set_zlabel(f'PC3 ({explained_variance_ratio_high_dim[2]*100:.1f}%)')

legend1 = ax.legend(*scatter.legend_elements(), title="Клас")
ax.add_artist(legend1)

plt.tight_layout()
plt.show()

# Визуализация на кумулативната дисперсия за 15D данни
X_full_high_dim_pca, full_components_high_dim, full_explained_variance_high_dim = my_pca(X_high_dim, n_components=X_high_dim.shape[1])
cumulative_variance_ratio_high_dim = np.cumsum(full_explained_variance_high_dim) / np.sum(full_explained_variance_high_dim)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance_ratio_high_dim) + 1), cumulative_variance_ratio_high_dim, 'o-', markersize=8)
plt.grid(True)
plt.xlabel('Брой компоненти')
plt.ylabel('Кумулативна обяснена дисперсия')
plt.title('Кумулативна обяснена дисперсия за 15D данни')

for threshold in [0.7, 0.8, 0.9, 0.95]:
    components_needed = np.argmax(cumulative_variance_ratio_high_dim >= threshold) + 1
    plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=components_needed, color='r', linestyle='--', alpha=0.3)
    plt.text(components_needed + 0.1, threshold - 0.03, f'{components_needed} комп.: {threshold*100:.0f}%', 
             fontsize=9, color='red')

plt.ylim([0, 1.05])
plt.show()