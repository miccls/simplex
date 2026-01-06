
class SmallestSubscriptRule:
    
    @staticmethod
    def pick_entering_index(reduced_costs):
        for i, cost in enumerate(reduced_costs):
            if (cost < 0):
                return i
        raise Exception("Pivoting strategy assumes some reduced cost is negative.")