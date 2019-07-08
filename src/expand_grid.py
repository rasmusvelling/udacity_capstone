import itertools

def expand_grid(**kwargs):

    holder=[]
    names=[]

    for key, value in kwargs.items():
        holder.append(value)
        names.append(key)

    product = list(itertools.product(*holder))

    output = {names[i]: [x[i] for x in product] for i in range(len(kwargs))}

    return output
