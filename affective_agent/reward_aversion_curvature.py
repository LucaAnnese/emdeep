import numpy as np
import matplotlib.pyplot as plt



def wundt(G, ha= 2, hr= 1, ca= 15, cr= 15, ga= 12, gr= 5):
    """
    Compute the valence level wrt the arousal level (information gain) using the Wundt curvature model
    :param G: information gain
    :param ha: maxima of negative valence
    :param hr: maxima of positive valence
    :param ca: gradient of negative valence
    :param cr: gradient of positive valence
    :param ga: activation threshold of aversion
    :param gr: activation threshold of reward
    :return: valence level
    """

    assert gr < ga, "gr must be less than ga"

    #assert G >= 0, "Information gain must be greater or equal to 0"

    reward = hr / (1 + np.exp(-cr*G + gr))
    aversion = -ha / (1 + np.exp(-ca*G + ga))
    valence = reward + aversion

    return valence, reward, aversion



x = np.linspace(0, 1, 100)

y, reward, aversion = wundt(x)
plt.plot(x, y)
plt.plot(x, reward)
plt.plot(x, aversion)
plt.ylim(-1, 1)
plt.title("Wundt Mapping of arousal in [-1,1]")
plt.xlabel("Arousal in [0,1]")
plt.ylabel("Arousal in [-1,1]")
plt.legend(["Arousal", "Reward", "Aversion"])