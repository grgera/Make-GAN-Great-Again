import os
import glob


def dashjoin(values):
    return '-'.join([str(v) for v in values])

def ulinejoin(values):
    return '_'.join([str(v) for v in values])

def dotjoin(values):
    return '.'.join([str(v) for v in values])

class PathBuilder:
    def __init__(self, train_size=0, test_size=0, model_params={}, training_params={}, pretrain_params={}, model_file_suffix='trc', path=None, debug=False, no_save=False):
        """
        Inputs:
            - model_params, training_params: {'model_name': {'param_name': param_value}}
        """
        if path is not None:
            dir_path = os.path.dirname(path)
            train_size, test_size, model_params, training_params, pretrain_params = self.unparse_path(dir_path)

        self.train_size = train_size
        self.test_size = test_size
        self.model_params = model_params
        self.training_params = training_params
        self.pretrain_params = pretrain_params
        self.model_file_suffix = model_file_suffix
        self.model_path_prefix = None
        self.model_pretrain_path_prefix = None
        self.has_pretrained_models = False
        self.has_trained_models = False
        self.debug = debug
        self.no_save = no_save

        self.update_model_path_prefix()
        if not debug and not no_save:
            self.mkdir()

    def model_path(self, model_name):
        return os.path.join(self.model_path_prefix, dotjoin([model_name, self.model_file_suffix]))
    
    def model_pretrain_path(self, model_name):
        return os.path.join(self.model_pretrain_path_prefix, dotjoin([model_name, self.model_file_suffix]))

    def model_eval_output_path(self, pretrain=False, no_score=False):
        model_path_prefix = self.model_path_prefix if not pretrain else self.model_pretrain_path_prefix
        model_path_suffix = os.path.normpath(model_path_prefix).split(os.sep)[1:]
        filename = 'results.tsv' if not no_score else 'results_raw.tsv'
        return os.path.join('output', *model_path_suffix, filename)

    def increment_training_params(self, training_params, pretrain=False):
        for model, params in training_params.items():
            for pk, pv in params.items():
                if not pretrain:
                    self.training_params[model][pk] += pv
                else:
                    self.pretrain_params[model][pk] += pv

        self.update_model_path_prefix()

    def whole_string(self):
        return '_'.join(self.model_pretrain_path_prefix.split('/'))

    def update_model_path_prefix(self):
        train_test_set = f'{self.train_size}_{self.test_size}'
        model_params = [ulinejoin([model] + [dashjoin([pk, pv]) for pk, pv in params.items()]) for model, params in self.model_params.items()]
        training_params = [ulinejoin([model] + [dashjoin([pk, pv]) for pk, pv in params.items()]) for model, params in self.training_params.items()]
        pretrain_params = [ulinejoin([model] + [dashjoin([pk, pv]) for pk, pv in params.items()]) for model, params in self.pretrain_params.items()]

        # Main model
        old_prefix = self.model_path_prefix
        self.model_path_prefix = os.path.join('model', train_test_set, ulinejoin(model_params), ulinejoin(training_params))

        if old_prefix is not None and os.path.exists(old_prefix):
            os.rename(old_prefix, self.model_path_prefix)
        
        # Pretrained model
        old_prefix = self.model_pretrain_path_prefix
        self.model_pretrain_path_prefix = os.path.join(self.model_path_prefix, 'pretrain', ulinejoin(pretrain_params))

        if old_prefix is not None and os.path.exists(old_prefix):
            os.rename(old_prefix, self.model_pretrain_path_prefix)
    
    def mkdir(self):
        # If a model with the same params already exists, the paths may only differ in the `iter` param for adversarial training
        # Make a wildcard path for this so that the existence of a pretrained model can be detected
        start = self.model_path_prefix.find('iter-')
        end = self.model_path_prefix.find('_', start)

        wildcard_pretrain_path = self.model_pretrain_path_prefix
        wildcard_pretrain_path = wildcard_pretrain_path.replace(wildcard_pretrain_path[start:end], 'iter-*')
        wildcard_pretrain_path = os.path.join(wildcard_pretrain_path, f'*.{self.model_file_suffix}')
        
        wildcard_path = self.model_path_prefix
        wildcard_path = wildcard_path.replace(wildcard_path[start:end], 'iter-*')
        wildcard_path = os.path.join(wildcard_path, f'*.{self.model_file_suffix}')

        pretrain_paths = glob.glob(wildcard_pretrain_path)
        paths = glob.glob(wildcard_path)
        if len(pretrain_paths) == 0:
            os.makedirs(self.model_pretrain_path_prefix)
            self.has_pretrained_models = False
            self.has_trained_models = False 
        else:
            path = os.path.dirname(pretrain_paths[0]) 
            self.train_size, self.test_size, self.model_params, self.training_params, self.pretrain_params = self.unparse_path(path)
            self.update_model_path_prefix()
            self.has_pretrained_models = True
            self.has_trained_models = len(paths) != 0

    def unparse_path(self, path):
        _, train_test_set, model_params_str, training_params_str, _, pretrain_params_str = os.path.normpath(path).split(os.path.sep)

        train_size, test_size = [int(s) for s in train_test_set.split('_')]
        model_params = self.unparse_params_str(model_params_str)
        training_params = self.unparse_params_str(training_params_str)
        pretrain_params = self.unparse_params_str(pretrain_params_str)

        return train_size, test_size, model_params, training_params, pretrain_params
        
    def unparse_params_str(self, params_str):
        params_str_list = params_str.split('_')
        params = {} 
        for i, s in enumerate(params_str_list):
            if s.find('-') == -1:
                model = s
                params[model] = {}
            else:
                pk, pv = s.split('-')
                pv = float(pv) if pv.find('.') != -1 else int(pv)
                params[model][pk] = pv
        return params

    @classmethod
    def ensure(cls, filepath):
        """
        Ensures the directory exists for the file.
        """
        dirname = os.path.dirname(filepath)
        if not os.path.exists(dirname):
            os.makedirs(dirname)