
import numpy as np

class d:
    def __iter__(self):
        self.n = -1
        return self

    def __next__(self):
        if self.n < self.nMax-1:
            self.n += 1
            return self
        else:
            raise StopIteration

    def runspark(*sparkArgs, **sparkKwargs):
        def decorator(worker):
            #define a new calculate method
            def sparkcalculation(self, *args, **kwargs):
                print('print args')
                for a in args:
                    print(a)

                if sparkKwargs.size==0:
                    return worker

                print('print kwargs', sparkKwargs.keys())
                for a in sparkKwargs:
                    print(a)

                sparkSession = sparkKwargs['sparkSession']
                partitions = sparkKwargs['partitions']

                data1 = args[0]
                data2 = args[1]
                data3 = args[2]

                print(f'running a heavier calculation with {sparkSession} and partitions {partitions}')
                return data1+data2+data3+1000

            #replace the calculate with newcalculate
            worker.calculate = sparkcalculation
            #return the modified class
            return worker
        return decorator




@d.runspark(sparkSession='a fake section', partitions=2)
class Worker:
    def __init__(self, name):
        self.name = name

    def calculate(self, data, data2, data3):
        print('running a calculation')
        return data+data2+data3

# enableSpark()
obj = Worker('Pencil Programmer')
res = obj.calculate(1, 2, 3)
print(f'Result: {res}')
