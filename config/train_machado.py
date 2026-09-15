# treina um mini GPT com BPE proprio, no corpus de ficcao do Machado de Assis
# bom para debugar e brincar em notebooks/laptops

out_dir = 'out-machado'
eval_interval = 250 # mantem frequente porque vamos overfitar
eval_iters = 200
log_interval = 10 # nao imprime com frequencia demais

# esperamos overfitar nesse dataset pequeno, entao so salva quando o val melhorar
always_save_checkpoint = False

wandb_log = False # sobrescreva via linha de comando se quiser
wandb_project = 'machado'
wandb_run_name = 'mini-gpt-machado'

dataset = 'machado'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # contexto de ate 256 tokens anteriores

# baby GPT :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # com redes pequenas da pra ir um pouco mais alto
max_iters = 5000
lr_decay_iters = 5000 # geralmente igual ao max_iters
min_lr = 1e-4 # geralmente learning_rate / 10
beta2 = 0.99 # um pouco maior porque o numero de tokens por iteracao e pequeno

warmup_iters = 100 # nao é super necessario, potencialmente

# se rodar em notebook/CPU, tambem adicione:
# device = 'cpu'
# compile = False