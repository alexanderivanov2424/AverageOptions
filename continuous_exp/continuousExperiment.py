
class Experiment:
    def __init__(self, exp_name):
        self.exp = {'exp_name': exp_name, 'env_name': "", 'methods': {}}
        self.exp_name = exp_name
    
    def set_env_name(self, env_name):
        self.exp["env_name"] = env_name

    def start_method(self, method):
        if not method in self.exp['methods']:
            self.exp['methods'][method] = {'avg_planning_steps': {}, 'runs': []}

    def log_average_planning_steps(self, method, num_options, avg_planning_steps):
        self.exp['methods'][method]['avg_planning_steps'][num_options] = avg_planning_steps

    def get_average_planning_steps(self, method, num_options):
        return self.exp['methods'][method]['avg_planning_steps'][num_options]

    def get_options_and_average_planning_steps(self, method):
        num_options = list(self.exp['methods'][method]['avg_planning_steps'].keys())
        average_planning_steps = [self.get_average_planning_steps(method, x) for x in num_options]
        return num_options, average_planning_steps

    def log_planning_run(self, method, num_ops, s_pos, g_pos, distance, planning_steps, options_used):
        d = {'num_options': num_ops, 'start_pos': s_pos, 'goal_pos':g_pos, 'distance':distance, 'planning_steps':planning_steps, 'options_used':options_used}
        self.exp['methods'][method]['runs'].append(d)

    # def saveToFile(self, directory="./continuous_exp/experiments/"):
    #     path = os.path.join(directory, self.exp_name) + ".json"
    #     with open(path, "w") as file:
    #         json.dump(self.exp, file)

    # def loadFromFile(self, directory="./continuous_exp/experiments/"):
    #     path = os.path.join(directory, self.exp_name) + ".json"
    #     with open(path, "r") as file:
    #         self.exp = json.load(file)
    #     self.exp_name = self.exp['exp_name']