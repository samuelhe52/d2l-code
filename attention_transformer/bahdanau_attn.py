import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Optional, Tuple
from utils.attn import AdditiveAttention
from utils.data.mt_data import GerEngDataset, eval_translations, mt_dataloader
from utils.enc_dec import Encoder, Decoder, EncoderDecoder
from utils.io import load_model
from utils.training import TrainingLogger, TrainingConfig
from utils.training import Seq2SeqTrainer

class BahdanauEncoder(Encoder):
    """
    Sequence-to-Sequence Encoder, implemented using GRU.
    """
    def __init__(self, vocab_size: int, embed_size: int,
                 num_hiddens: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout, batch_first=False)
        
    def forward(self, X: Tensor, *args) -> Tuple[Tensor, Tensor]:
        # X shape: (batch_size, seq_len)
        embs = self.embedding(X.T.to(torch.long))
        # embs shape: (seq_len, batch_size, embed_size)
        outputs, state = self.rnn(embs)
        # outputs shape: (seq_len, batch_size, num_hiddens)
        # state shape: (num_layers, batch_size, num_hiddens)
        return outputs, state

class BahdanauDecoder(Decoder):
    """
    Sequence-to-Sequence Decoder with Bahdanau Attention, implemented using GRU.
    """
    def __init__(self, vocab_size: int, embed_size: int,
                 num_hiddens: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self._vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = AdditiveAttention(
            query_size=num_hiddens,
            key_size=num_hiddens,
            num_hiddens=num_hiddens, dropout=dropout)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens,
                          num_layers, dropout=dropout, batch_first=False)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        
    @property
    def vocab_size(self) -> int: 
        """Return the size of the target vocabulary."""
        return self._vocab_size
    
        
    def preprocess_state(self, enc_outputs, *args):
        outputs, hidden_state = enc_outputs
        src_valid_len = args[0] if len(args) > 0 else None
        return outputs, hidden_state, src_valid_len
    
    def forward(self, X: Tensor, state: Tuple[Tensor, Tensor, Tensor | None]) \
        -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor | None]]:
        embs = self.embedding(X.T.to(torch.long))
        # embs shape: (seq_len, batch_size, embed_size)
        # enc_outputs shape: (seq_len, batch_size, num_hiddens)
        enc_outputs, hidden_state, src_valid_len = state
        enc_kv = (
            enc_outputs.permute(1, 0, 2)
            if enc_outputs.shape[0] != hidden_state.shape[1]
            else enc_outputs
        )
        # enc_kv shape: (batch_size, seq_len, num_hiddens)
        outputs = []

        for emb in embs:
            # emb shape: (batch_size, embed_size)
            # num_query = 1
            query = hidden_state[-1].unsqueeze(1)  # Shape: (batch_size, 1, num_hiddens)
            # enc_kv is both keys and values
            context = self.attention(query, enc_kv, enc_kv, src_valid_len)
            # context shape: (batch_size, 1, num_hiddens)
            rnn_inputs = torch.cat((emb.unsqueeze(1), context), dim=-1).transpose(0, 1)
            # rnn_inputs shape: (1, batch_size, embed_size + num_hiddens)
            out, hidden_state = self.rnn(rnn_inputs, hidden_state)
            # out shape: (1, batch_size, num_hiddens)
            outputs.append(out.squeeze(0))

        outputs = torch.stack(outputs, dim=0)
        # outputs shape: (seq_len, batch_size, num_hiddens)
        outputs = self.dense(outputs).permute(1, 2, 0)
        # outputs shape: (batch_size, vocab_size, seq_len)
        return outputs, (enc_outputs, hidden_state, src_valid_len)
    

class Seq2Seq(EncoderDecoder):
    """
    Sequence-to-Sequence model.
    """
    def __init__(self,
                 encoder: BahdanauEncoder,
                 decoder: BahdanauDecoder,
                 pad_token_index: int,
                 eos_token_index: int | None = None):
        super().__init__(encoder, decoder)
        self.pad_token_index = pad_token_index
        self.eos_token_index = eos_token_index
        

if __name__ == "__main__":
    hparams = {
        'seq_len': 25,
        'batch_size': 128,
        'num_epochs': 15,
        'lr': 2e-3,
        'grad_clip': 1.0,
        'embed_size': 256,
        'num_hiddens': 512,
        'num_layers': 2,
        'dropout': 0.2,
    }
    
    data = GerEngDataset(
        seq_len=hparams['seq_len'],
        token_min_freq=5,
        # total_samples=20000,
    )
    train_loader = mt_dataloader(
        data, batch_size=hparams['batch_size'], train=True
    )
    val_loader = mt_dataloader(
        data, batch_size=hparams['batch_size'], train=False
    )
    
    shared = {
        'embed_size': hparams['embed_size'],
        'num_hiddens': hparams['num_hiddens'],
        'num_layers': hparams['num_layers'],
        'dropout': hparams['dropout'],
    }
    
    encoder = BahdanauEncoder(vocab_size=len(data.src_vocab), **shared)
    decoder = BahdanauDecoder(vocab_size=len(data.tgt_vocab), **shared)
    model = Seq2Seq(
        encoder=encoder, decoder=decoder,
        pad_token_index=data.tgt_vocab['<pad>'],
        eos_token_index=data.tgt_vocab['<eos>'],
    )
    
    logger = TrainingLogger(
        log_path='logs/bahdanau_mt_gereng.json',
        hparams=hparams
    )
    
    optim = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    
    config = TrainingConfig(
        num_epochs=hparams['num_epochs'],
        lr=hparams['lr'],
        grad_clip=hparams['grad_clip'],
        optimizer=optim,
        save_path='./models/bahdanau_mt_gereng',
        logger=logger,
        device=torch.device('mps'),
    )
    
    def init_seq2seq(m):
        """Initialize weights for sequence-to-sequence learning."""
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    model.apply(init_seq2seq)
    trainer = Seq2SeqTrainer(model, train_loader, val_loader, config)
    trainer.train()
    logger.summary()
    
    model: Seq2Seq = load_model('./models/seq2seq_mt_gereng',
                                model, device=torch.device('cpu'))
    engs, des = data.test_sentences
    preds, _ = model.generate(
        data.build(engs, des),
        torch.device('cpu'),
        max_len=50,
        decode_strategy='beam',
        beam_size=4)
    preds = preds.cpu().tolist()
    
    eval_translations(srcs=engs, dsts=des, preds=preds, data=data)
    