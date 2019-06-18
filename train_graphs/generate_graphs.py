

from cdt.generators import AcyclicGraphGenerator

for mechanism in ['polynomial','NN','sigmoid_add', 'sigmoid_mix',
                  'gp_add', 'gp_mix', 'linear']:#  'linear', 'sigmoid_add', 'sigmoid_mix',
    a = AcyclicGraphGenerator(mechanism, nodes=20)
    a.generate()
    a.to_csv("-".join([str(j) for j in [mechanism,'train', str(20)]]), index=False)
