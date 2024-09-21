<center><h1>Aprendendo Matemática com Python</h1></center>
<center><h2>Curso de Extensão</h2></center>
<center><h3>Fernando Deeke Sasse</h3></center>
<center><h3>CCT - UDESC</h3></center>
<center><h2>NumPy: Vetores Espaciais, Matrizes e Sistemas Lineares</h2></center>

### 1. Introdução

Veremos neste notebook como realizar operações algébricas envolvendo 3-vetores e matrizes. MOstraremos algumas aplicações elementares em álgebra linear, embora o tratamento aprofundado desse tópico esteja reservado para uma aula futura. 

### 2. Operações com 3-vetores espaciais

Consideremos os vetores $\mathbf{u}=(3,1,2)$ e $\mathbf{v}=(-1,2,-1)$. O produto escalar $\mathbf{u}\cdot \mathbf{v}$ pode ser feito na biblioteca NumPy da seguinte forma: 


```python
import numpy as np
u = np.array([3,1,2])
v = np.array([-1,2,-1])
np.dot(u,v)
```




    -3



Por outro lado, o produto vetorial $\mathbf{u} \times \mathbf{v}$ é dado por 


```python
np.cross(u,v)
```




    array([-5,  1,  7])



A norma de Frobenius (ou euclidiana) de um vetor  $\mathbf{u}=(u_1,\ldots,u_n)$ é definida por

\[
|\mathbf{u}|= \sqrt{\Sigma_{i=1}^n|u_i|^2}
\]
No Python: 


```python
u = np.array([4,-6,1])
NE = np.linalg.norm(u)
NE
```




    7.280109889280518



De fato, 


```python
np.sqrt(np.sum(u**2))
```




    7.280109889280518



A chamada "norma infinito" de um vetor é definida como sendo a maior magnitude entre as componentes:  

\[
|\mathbf{u}|_{\infty}=\max_i\{{|u_i|},i=1,\ldots,n\}
\]

Ela é usada em álgebra linear como uma medida do tamanho de um vetor de alta dimensão, uma vez que a norma euclideano tem alto custo computacional. Por exemplo,



```python
u = np.array([4,-6,1,8,-21,2,16])
```


```python
NI = np.linalg.norm(u,ord=np.inf)
NI
```




    21.0



**Exemplo 2.1**. Sejam os vetores $\mathbf{u}=(1,4,-1)$ e $\mathbf{v}=(1,3,-5)$. 

(i) Calcule a projeção do vetor $\mathbf{u}$ na direção de $\mathbf{v}$. 

Devemos calcular a expressão 
\[
\mathbf{u}_{\mathbf{v}}=\left(\frac{\mathbf{u}\cdot\mathbf{v}}{|\mathbf{v}|^2}\right)\mathbf{v}
\]

No Python, 


```python
u = np.array([1,4,-1])
v = np.array([1,3,-5])
uv = (np.dot(u,v)/np.dot(u,v))*v
uv
```




    array([ 1.,  3., -5.])



(ii) Calcule o ângulo $\theta$ entre $\mathbf{u}$ e $\mathbf{v}$. 

Para isso podemos usar a relação

\[
\mathbf{u}\cdot \mathbf{v}=|\mathbf{u}||\mathbf{v}|\cos\theta\,.
\]
Em Python,


```python
c = np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v)
theta = np.arccos(c)
theta
```




    0.771110504762802



Em graus:


```python
theta_deg = np.degrees(theta)
theta_deg
```




    44.18137746111112



De fato, 


```python
theta*180/np.pi
```




    44.18137746111112



**Exemplo 2.2** Determine a área do triângulo com vértices $(-2,3,1)$, $(3,2,4)$ e $(2,1,8)$.

Devemos lembrar que a área do triângulo $ABC$ pode ser calculada pela fórmula

\[
A = \frac{1}{2}|\mathbf{AB}\times \mathbf{AC}|,
\]

sendo $\mathbf{AB}$ ($\mathbf{AC}$) o vetor com origem no ponto $A$ e extremidade no ponto $B$ ($C$). Façamos o cálculo em Python:


```python
A = np.array([-2,3,1])
B = np.array([3,2,4])
C = np.array([2,1,8])
AB = B-C
AC = C-A
```

A área é então dada por


```python
Area = np.linalg.norm(np.cross(AB,AC))/2
Area
```




    11.895377253370318



**Exemplo 2.3**

Determine o volume do tetraedro com vértices $A =(2,3,1)$, $B = (3,-2,4)$, $C = (2,1,8)$ e $D =(-3,1,-2)$.

O volume do tetraedro é 1/6 do volume do paralepípedo definido pelos vértices acima. O volume do paralelepípedo é dado pela magnitude do produto triplo do vetores definidos pelos vértices $A$, $B$, $C$ e $D$:

\[
V_t=\frac{1}{6}|\mathbf{AB}\cdot(\mathbf{AC}\times \mathbf{AD})|
\]

Em Python temos:


```python
A = np.array([2,3,1])
B = np.array([3,-2,4])
C = np.array([2,1,8])
D = np.array([-3,1,-2])
AB = B-A
AC = C-A
AD = D-A
Vt=1/6*np.dot(AB, np.cross(AC,AD))
Vt
```




    27.5



### 3. Matrizes no Numpy

Matrizes são arrays 2-dimensionais. Por exemplo, seja a matriz $3 \times 4$:

\[
M=\begin{bmatrix}
-1& 5 & 7 & 4\\
2 & 3 & 1 & 7\\
8 & 9 & 6 & 3
\end{bmatrix}
\]

Em Python, 


```python
M=np.array([[-1,5,7,4],[2,3,1,7],[8,9,6,3]])
M
```




    array([[-1,  5,  7,  4],
           [ 2,  3,  1,  7],
           [ 8,  9,  6,  3]])



A dimensão de $M$ é dada por 


```python
M.ndim
```




    2



Sua forma é: 


```python
M.shape
```




    (3, 4)



O número total de elementos é dado por


```python
M.size
```




    12



### 4. Fatiamento de matrizes

Podemos selecionar componentes de $M$ da forma usual. Por exemplo, o elemento da primeira linha e segunda coluna é dado por


```python
M[0,1]
```




    5



Selecionemos a primeira coluna de $M$ para gerar um array 1D: 


```python
col1=M[:,0]
col1
```




    array([-1,  2,  8])



Podemos forma submatrizes. Por exemplo, selecionemos a submatriz 

\[
\begin{bmatrix}
 1 & 7\\
6 & 3
\end{bmatrix}
\]



```python
M[1:3,2:4]
```




    array([[1, 7],
           [6, 3]])




```python
A = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10],[5, 1, 9, 2, -4]])
A
```




    array([[ 1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10],
           [ 5,  1,  9,  2, -4]])




```python
A[0:2, 1:4]
```




    array([[2, 3, 4],
           [7, 8, 9]])



### 5. Matrizes especiais


```python
Matrizes especiais estão pré-definidas. Por exemplo
```


```python
I4=np.eye(4) # matriz identidade
I4
```




    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])




```python
B3 = np.ones((3,3)) # matriz com 1 em todas as componentes
B3
```




    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]])




```python
Z3 = np.zeros((3,3))
Z3
```




    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])



Podemos definir matrizes com componentes aleatórias com diferentes distribuições:


```python
# Distribuição uniforme
np.random.seed(122)
m=3
n=4
A_uniforme = np.random.rand(m, n)
A_uniforme
```




    array([[0.15699184, 0.70221004, 0.26147827, 0.45171291],
           [0.40806526, 0.61154121, 0.5830806 , 0.21450887],
           [0.1887731 , 0.70209797, 0.47081767, 0.48505287]])




```python
# Distribuição normal padrão
np.random.seed(122)
m=3
n=4
A_normal = np.random.randn(m, n)
A_normal
```




    array([[ 0.48468014, -0.82216117, -0.33674179, -1.66338939],
           [ 1.71928769, -1.41707497, -1.38444536,  0.40288662],
           [ 0.59476914, -0.91593276, -1.50496569, -2.93824983]])




```python
# Distribuição normal qualquer
np.random.seed(122)
m=3
n=4
mean = 10
std_dev = 2
A_normal2 = np.random.normal(mean, std_dev, size=(m, n))
A_normal2
```




    array([[10.96936028,  8.35567767,  9.32651641,  6.67322122],
           [13.43857538,  7.16585006,  7.23110927, 10.80577325],
           [11.18953828,  8.16813447,  6.99006863,  4.12350034]])



Podemos definir matrizes com componentes aleatórias inteiras:


```python
# Inteiros de 2 a 30 (exclusivo)
np.random.seed(122)
m=3
n=4
A_int = np.random.randint(2,30, size=(m, n))
A_int
```




    array([[17, 28, 12, 24],
           [14, 18, 14, 17],
           [29,  4, 17, 24]])



### 6. Definindo matrizes por uma lei de formação

Podemos definir uma matriz de acordo com uma lei de formação para seus elementos $a_{ij}$. O modo mais imediato, mas não eficiente,  é o seguinte:


```python
n = 4
# Inicializa a matriz vazia
A = np.zeros((n, n), dtype=float)

# Define a matriz usando loops
for i in range(n):
    for j in range(n):
        A[i, j] = i/(1+2*j) + j
A
```




    array([[0.        , 1.        , 2.        , 3.        ],
           [1.        , 1.33333333, 2.2       , 3.14285714],
           [2.        , 1.66666667, 2.4       , 3.28571429],
           [3.        , 2.        , 2.6       , 3.42857143]])



Podemos usar o comando `fromfunction` do Numpy:


```python
n = 4  # tamanho da matriz
A = np.fromfunction(lambda i, j: i/(1+2*j) + j, (n, n), dtype=int)
A
```




    array([[0.        , 1.        , 2.        , 3.        ],
           [1.        , 1.33333333, 2.2       , 3.14285714],
           [2.        , 1.66666667, 2.4       , 3.28571429],
           [3.        , 2.        , 2.6       , 3.42857143]])



Outro modo de definir uma matriz é a seguinte:


```python
n = 4
A = np.array([[i/(j+1) + j for j in range(0,n)] for i in range(0,n)])
A
```




    array([[0.        , 1.        , 2.        , 3.        ],
           [1.        , 1.5       , 2.33333333, 3.25      ],
           [2.        , 2.        , 2.66666667, 3.5       ],
           [3.        , 2.5       , 3.        , 3.75      ]])



### 7. Operações com matrizes

Seja a matrix  $M$ dada por:  


```python
M=np.array([[-1,3,7,4],[2,3,1,7],[6,9,1,3]])
print(M)
```

    [[-1  3  7  4]
     [ 2  3  1  7]
     [ 6  9  1  3]]


Operações sobre arrays são feitas elemento a elemento. Por exemplo.


```python
M**2 #Eleva cada componente ao quadrado
```




    array([[ 1,  9, 49, 16],
           [ 4,  9,  1, 49],
           [36, 81,  1,  9]])




```python
np.sin(M) #Calcula o seno de cada componente
```




    array([[-0.84147098,  0.14112001,  0.6569866 , -0.7568025 ],
           [ 0.90929743,  0.14112001,  0.84147098,  0.6569866 ],
           [-0.2794155 ,  0.41211849,  0.84147098,  0.14112001]])



Para elevar a matriz quadrado fazemos


```python
A = np.array([[2,3,4],[-5,3,1],[3,5,6]])
A
```




    array([[ 2,  3,  4],
           [-5,  3,  1],
           [ 3,  5,  6]])




```python
A@A
```




    array([[  1,  35,  35],
           [-22,  -1, -11],
           [ -1,  54,  53]])



Façamos uma multiplicação matricial. Definimos uma matriz $1 \times 3$(note os dois colchetes, que caracteriza o 2-array):


```python
B = np.array([[2,3,5,7]])
B
```




    array([[2, 3, 5, 7]])



e obtemos sua transposta:


```python
Bt = np.transpose(B)
Bt
```




    array([[2],
           [3],
           [5],
           [7]])



ou


```python
B.T
```




    array([[2],
           [3],
           [5],
           [7]])




```python
M@Bt
```




    array([[70],
           [67],
           [65]])



Note acima que 


```python
B.shape
```




    (1, 4)



Para calcular a inversa de uma matriz procedemos do seguinte modo:


```python
A = np.array([[2,-3],[4,6]])
np.linalg.inv(A)
```




    array([[ 0.25      ,  0.125     ],
           [-0.16666667,  0.08333333]])



Se


```python
M = np.array([[13,-8,6,4],[3,1,6,5],[1,0,3,2],[1,3,8,-2]])
M
```




    array([[13, -8,  6,  4],
           [ 3,  1,  6,  5],
           [ 1,  0,  3,  2],
           [ 1,  3,  8, -2]])



O traço da matriz $M$ (soma dos elementos da diagonal) é dado por 


```python
np.trace(M)
```




    15



O determinante é dado por: 


```python
np.linalg.det(M)
```




    -476.9999999999997



### 8. Normas matriciais

Normas matriciais são funções que atribuem um número real não negativo a uma matriz, quantificando o "tamanho" ou "magnitude" da matriz de uma maneira específica. Assim como as normas vetoriais medem o comprimento ou a magnitude de um vetor, as normas matriciais são usadas para medir determinados tipos de magnitude de uma matriz.

As normas matriciais são amplamente utilizadas em álgebra linear, análise numérica e outras áreas matemáticas para analisar a estabilidade, sensibilidade e comportamento de operações envolvendo matrizes.

Uma função $ \|\cdot\|: \mathbb{R}^{m \times n} \rightarrow \mathbb{R} $ é considerada uma norma matricial se satisfizer as seguintes condições:

1. Subaditividade (Desigualdade Triangular):
   \[
   \|A + B\| \leq \|A\| + \|B\| \quad \text{para todas as matrizes } A \text{ e } B \text{ de mesma dimensão}.
   \]

2. Multiplicação Escalar:
   \[
   \|\alpha A\| = |\alpha| \cdot \|A\| \quad \text{para todo escalar } \alpha \text{ e matriz } A.
   \]

3. Não-Negatividade e Definitividade:
  \[
   \|A\| \geq 0 \quad \text{e} \quad \|A\| = 0 \text{ se e somente se } A = 0.
   \]
   
4. Compatibilidade com a Multiplicação Matricial:
   \[
   \|AB\| \leq \|A\| \cdot \|B\| \quad \text{para todas as matrizes } A \text{ e } B \text{ onde o produto } AB \text{ é definido}.
   \]



Examinemos os principais tipos de normas matriciais:

- Norma 1 (Norma da Coluna Máxima, $ p = 1 $):
  \[
  \|A\|_1 = \max_{1 \leq j \leq n} \sum_{i=1}^{m} |a_{ij}|
  \]
  A norma 1 mede a maior soma absoluta das colunas da matriz.
  
  Em Python, 


```python
import numpy as np
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
# Norma 1
norm1 = np.linalg.norm(A, 1)
norm1
```




    18.0



- Norma 2 (Norma Espectral, $ p = 2 $):
  \[
  \|A\|_2 = \sqrt{\lambda_{\text{max}}}\,,
  \]
  sendo  $ \lambda_{\text{max}}$ maior autovalor de $ A^{\dagger}A $. Aqui $A^{\dagger}$ é a matriz hermitiana conjugada de $A$. Se $A$ for real, $A^{\dagger}=A^T$ Em Python, 


```python
norm2 = np.linalg.norm(A, 2)
norm2
```




    16.84810335261421



- Norma Infinito (Norma da Linha Máxima, $ p = \infty $):
  \[
  \|A\|_\infty = \max_{1 \leq i \leq m} \sum_{j=1}^{n} |a_{ij}|
  \]
  
  A norma infinito mede a maior soma absoluta das linhas da matriz. Em Python,


```python
norm_inf = np.linalg.norm(A, np.inf)
norm_inf
```




    24.0



- Norma de Frobenius: 

\[
\|A\|_F = \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} |a_{ij}|^2}
\]

Esta norma é equivalente à norma 2 para o vetor de elementos da matriz, sendo a raiz quadrada da soma dos quadrados de todos os elementos da matriz. Em Python, 


```python
norm_fro = np.linalg.norm(A, 'fro')
norm_fro
```




    16.881943016134134



Normas matriciais podem ser usadas para estabelecer critérios de parada em algoritmos numéricos, análise de estabilidade de equações diferenciais, etc. Uma aplicação que exemplificaremos aqui é o cálculo do número de condionamento de uma matriz. 

### 9. Número de condicionamento de uma matriz

O número de condicionamento de uma matriz $ A $ é definido como:

\[
\kappa(A) = \|A\| \cdot \|A^{-1}\|\,,
\]

sendo $ \|A\| $ a norma da matriz $ A$, e $ \|A^{-1}\| $ é a norma de sua inversa.  

Esta é uma medida que indica a sensibilidade da solução de um sistema linear em relação a mudanças nos dados de entrada. Um número de condicionamento elevado sugere que a matriz pode causar problemas numéricos, como soluções instáveis ou imprecisas.

O número de condicionamento frequentemente utiliza a norma 2 (norma espectral), que está relacionada aos autovalores da matriz. Este é o default em Python:


```python
import numpy as np
A = np.array([
    [2, 3, -1],
    [4, 1, 2],
    [-2, 5, 2]
])
# Norma 2
cond_num = np.linalg.cond(A)
cond_num
```




    2.5141162817681



Usando outras normas:


```python
# Norma 1
cond_num1 = np.linalg.cond(A, p=1)
cond_num1
```




    5.108108108108108




```python
# Norma infinito
cond_num_inf = np.linalg.cond(A, p=np.inf)
cond_num_inf
```




    5.837837837837839




```python
# Norma Frobenius
cond_num_fro = np.linalg.cond(A, p='fro')
cond_num_fro
```




    3.996163243915766



### 10. Autovalores e autovetores de uma matriz

Consideremos o problema de determinar os autovalores e autovetores da seguinte matriz 

\[
A = \begin{bmatrix}
2 & -1 & 0 \\
1 & 3 & -1 \\
0 & 2 & 1 \\
\end{bmatrix}
\]

Os autovalores $ \lambda $ são determinados pela equação característica:

\[
\det(A - \lambda I) = 0
\]

ou seja, 

\[
\det(A - \lambda I) = \left| \begin{array}{ccc}
2 - \lambda & -1 & 0 \\
1 & 3 - \lambda & -1 \\
0 & 2 & 1 - \lambda \\
\end{array} \right|=\lambda^3 - 6\lambda^2 + 14\lambda - 11 = 0\,.
\]

Utilizaremos o Python para encontrar as raízes da equação:


```python
import numpy as np
```


```python
# Coeficientes do polinômio característico
coeficientes = [1, -6, 14, -11]

# Encontrar as raízes da equação cúbica
autovalores = np.roots(coeficientes)
print("Autovalores:")
print(autovalores)
```

    Autovalores:
    [2.22669883+1.46771151j 2.22669883-1.46771151j 1.54660235+0.j        ]


Podemos utilizar o Python para calcular os autovalores e autovetores diretamente:


```python
A = np.array([[2, -1, 0],
              [1, 3, -1],
              [0, 2, 1]])

autovalores, autovetores = np.linalg.eig(A)
```


```python
autovalores
```




    array([2.22669883+1.46771151j, 2.22669883-1.46771151j,
           1.54660235+0.j        ])




```python
autovetores
```




    array([[ 0.36126918-0.21800183j,  0.36126918+0.21800183j,
            -0.50266229+0.j        ],
           [-0.4018631 -0.48081818j, -0.4018631 +0.48081818j,
            -0.2279059 +0.j        ],
           [-0.6551944 +0.j        , -0.6551944 -0.j        ,
            -0.83390019+0.j        ]])



Verifiquemos os resultados. Por exemplo, examinemos se para o primeiro autovalor $\lambda_1 $ e autovetor $\mathbf{v}_1$ acima temos $A\mathbf{v}_1=\lambda_1 \mathbf{v}_1$:


```python
# Selecionar o primeiro autovalor e autovetor
lambda_1 = autovalores[0]
v_1 = autovetores[:, 0]
```


```python
# Calcular A @ v_1
A @ v_1-lambda_1 * v_1
```




    array([ 6.66133815e-16+9.99200722e-16j, -9.99200722e-16+0.00000000e+00j,
           -2.22044605e-16-7.77156117e-16j])



Equivalentemente tal verificação poderia ser feita do seguinte modo:


```python
np.allclose(A @ v_1,lambda_1 * v_1)
```




    True



### 11. Sistemas Lineares 

Definiremos funções em Python que realizam as operações elementares sobre linhas de uma matriz que serão úteis para resolver sistemas de equações lineares ou simplesmente, sistemas lineares.  Um [sistema linear](https://pt.wikipedia.org/wiki/Sistema_de_equações_lineares) é um conjunto de equações lineares acopladas entre por meio de variáveis $x_1,\ldots, x_n$, da forma:

\[
\left\{
\begin{aligned}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n & = b_1 \\
a_{10}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n & = b_2 \\
& \vdots \\
a_{m1}x_1 + a_{m1}x_2 + \cdots + a_{mn}x_n & = b_m \\
\end{aligned}
\right.
\]

Em notação matricial o sistema linear de $n$ equações  e $n$ incógnitas é representado na forma 
 $A X= B$, sendo

\[
A = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & & & \vdots \\\
a_{n 1} & a_{n 2} & \cdots & a_{n n} \\
\end{bmatrix}
 \ \ , \ \
X = \begin{bmatrix}
x_1 \\\ x_2 \\ \vdots \\ x_n
\end{bmatrix}
 \ \ , \ \
b= \begin{bmatrix}
b_1 \\\ b_2 \\\ \vdots \\ b_n
\end{bmatrix} 
\]

**Exemplo.** Consideremos o seguinte sistema de equações lineares:

\[
\begin{cases}
2x + 3y - z = 5 \\
4x + y + 2z = 6 \\
-2x + 5y + 2z = -1\,.
\end{cases}
\]

Nosso objetivo é encontrar os valores de $ x $, $ y $ e $ z $ que satisfaçam simultaneamente todas as três equações.

Utilizaremos a biblioteca NumPy do Python para resolver o sistema.  Na forma matricial reescrevemos o sistema como:

\[
\begin{bmatrix}
2 & 3 & -1 \\
4 & 1 & 2 \\
-2 & 5 & 2
\end{bmatrix}
\begin{bmatrix}
x \\ y \\ z
\end{bmatrix}
=
\begin{bmatrix}
5 \\ 6 \\ -1
\end{bmatrix}\,,
\]

com:

\[
A = \begin{bmatrix} 2 & 3 & -1 \\ 4 & 1 & 2 \\ -2 & 5 & 2 \end{bmatrix}, \quad
X = \begin{bmatrix} x \\ y \\ z \end{bmatrix}, \quad
b = \begin{bmatrix} 5 \\ 6 \\ -1 \end{bmatrix}\,.
\]

Resolvamos este problema em Python,  usando comandos do Numpy:


```python
import numpy as np
```


```python
# Dados
A = np.array([
    [2, 3, -1],
    [4, 1, 2],
    [-2, 5, 2]
])
b = np.array([5, 6, -1])
```


```python
X = np.linalg.solve(A, b)
X
```




    array([ 1.52702703,  0.54054054, -0.32432432])



Podemos verificar o resultado:


```python
np.dot(A, X)-b
```




    array([ 0.0000000e+00,  0.0000000e+00, -4.4408921e-16])



### 12. Exercícios

**1.** Dados os vetores

\[
\mathbf{v}_1=(4,5,3)\,,\quad \mathbf{v}_2=(5,2,6)\,,
\]

determine o vetor projeção de $\mathbf{v}_1$ na direção de $\mathbf{v}_2$. 

**2.** Determine a área do triângulo de vértices $(3,4,2)$, $(-2,1,9)$ e $(-4,2,-1)$. 

**3.** Dados os vetores

\[
\mathbf{v}_1=(2,5,1)\,,\qquad \mathbf{v}_2=(-1,2,-3)\,,\quad \mathbf{v}_3=(-4,1,-7)\,,
\]

determine o volume do paralelepípedo definido por esses vetores. 

**4.** Verifique  que a norma de Frobenius da matriz
\[
M=\begin{bmatrix}
-3 & 5 & 3 \\
2 & -5 & 4 \\
1 & 9 & -2
\end{bmatrix}
\]
pode também ser dada por 

\[
||\mathbf{M}||_F=\sqrt{\left(\mbox{Tr}(M M^T\right)}\,.
\]

**5.** Sejam as matrizes 
\[ A = 
\begin{bmatrix}
1 & -2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{bmatrix}\,,\qquad
B = 
\begin{bmatrix}
9 & 4 & 6 \\
6 & 3 & 2 \\
1 & 2 & 3
\end{bmatrix}\,.
\]

(i) Determine $C=A^3-AB -4I$, sendo $I$ a matriz identidade $3 \times 3$.

(ii) Use a equação  $det(C-I\lambda)=0$ para encontrar os autovalores de $C$.

**6.** Seja o sistema linear

\[
\begin{bmatrix}
-1 & -2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{bmatrix}
\begin{bmatrix}
x  \\
y  \\
z 
\end{bmatrix}
=
\begin{bmatrix}
1  \\
4  \\
3\,.
\end{bmatrix}
\]

Resolva o sistema calculando explicitamente  a expressão: 

\[
\begin{bmatrix}
x  \\
y  \\
z 
\end{bmatrix}= \begin{bmatrix}
-1 & -2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{bmatrix}^{-1}\begin{bmatrix}
1  \\
4  \\
3\,.
\end{bmatrix}
\]

Note que este não é o método mais eficiente para resolver um sistema linear. 

**7.** As matrizes de Hilbert têm a seguinte lei de formação $a_{ij}=1/(i+j-1)$ para $i=1,\cdots,n$. Tais matrizes são muito usadas em testes de algoritmos numéricos, pelo fato de serem mal condicionadas. Construa uma função que calcula o número de condição (usando qualquer tipo de norma) de matrizes de Hilbert de ordem $3\times 3, 4\times 4, \ldots, 40\times 40$. Faça o mesmo para uma matriz de componentes aleatórias. Faça um gráfico simultâneo comparando a variação do número de condição para as duas matrizes. 

**8.** Determine os autovalores e autovetores da matriz

\[
A = \begin{bmatrix}
4 & 1 & 2 & 0 \\
1 & 3 & 0 & 1 \\
2 & 0 & 2 & 3 \\
0 & 1 & 3 & 4 \\
\end{bmatrix}\,.
\]



**8.** Resolva o sistema linear dado por 

\[
\left\{
\begin{aligned}
2x_1 + 3x_2 - x_3 + 4x_4 + x_5 & = 7 \\
-3x_1 + 5x_2 + 2x_3 - 6x_4 + 8x_5 & = -2 \\
4x_1 - x_2 + 7x_3 + 3x_4 - 5x_5 & = 10 \\
x_1 + 6x_2 - 3x_3 + 2x_4 + 4x_5 & = 5 \\
-2x_1 + 4x_2 + 5x_3 - x_4 + 6x_5 & = 1 \\
\end{aligned}
\right.
\]

e verifique o resultado. 
