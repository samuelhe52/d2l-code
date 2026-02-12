"""Datasets for training machine translation models."""
import math
from typing import Tuple, Optional
import re
import collections
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import hashlib
from pathlib import Path
import zipfile
import shutil

from .vocab import Vocab

class TatoebaDataset(Dataset):
    """
    Dataset for pairs of sentences in two languages,
    e.g., English and French, from the Tatoeba project.
    """
    
    def __init__(self,
                 seq_len: int = 10,
                 token_min_freq: int = 2,
                 *,
                 src_lang: str,
                 tgt_lang: str,
                 data_url: str,
                 data_file_name: str,
                 md5_hash: str,
                 data_root: str = './data',
                 total_samples: Optional[int] = None):
        """
        Initializes the dataset by downloading and processing the data.
        
        Args:
            seq_len (int): Maximum sequence length for sentences. Padding or truncation \
                will be applied as necessary.
            token_min_freq (int): Minimum frequency for tokens to be included in the vocabulary.
            src_lang (str): Source language code (e.g., 'en' for English).
            tgt_lang (str): Target language code (e.g., 'fr' for French).
            data_url (str): URL to download the dataset from. This should be a \
                URL from manythings.org/anki. The file is expected to be a zip archive.
                Unzipping is performed automatically.
            data_file_name (str): Name of the data file inside the zip archive.
            md5_hash (str): MD5 hash to verify the integrity of the downloaded file.
            data_root (str): Root directory to store the dataset.
            total_samples: Optional[int] = None: If specified, limits the dataset to this many samples.
        """
        self.seq_len = seq_len
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.data_root = data_root
        self.total_samples = total_samples
        
        raw = self._load_data(data_url, md5_hash, data_file_name)
        src_sentences, tgt_sentences = self._preprocess(raw, total_samples)
        src_tokenized = [sent.split(' ') + ['<eos>'] for sent in src_sentences]
        tgt_tokenized = [sent.split(' ') + ['<eos>'] for sent in tgt_sentences]

        # Pass nested lists directly — Vocab handles flattening internally
        self.src_vocab = Vocab(src_tokenized,
                               min_freq=token_min_freq,
                               reserved_tokens=['<pad>', '<bos>', '<eos>'])
        self.tgt_vocab = Vocab(tgt_tokenized,
                               min_freq=token_min_freq,
                               reserved_tokens=['<pad>', '<bos>', '<eos>'])

        tgt_array_full = self._build_arrays(
            tgt_tokenized, self.tgt_vocab, is_tgt=True)

        self.src_array, self.src_valid_len = self._build_arrays(
            src_tokenized, self.src_vocab, is_tgt=False,
            return_valid_len=True)
        self.tgt_array = tgt_array_full[:, :-1]
        self.label_array = tgt_array_full[:, 1:]
        
    def __len__(self) -> int:
        return len(self.src_array)
    
    def __getitem__(self, idx: int) -> Tuple[Tuple[Tensor, Tensor, Tensor],
                                             Tensor]:
        """
        Get the source and target sequences along with source valid lengths.
        
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[Tuple[Tensor, Tensor, Tensor], Tensor]: A tuple containing:
                - A tuple of source array, target array, source valid length.
                - A tensor label array.
        """
        return ((self.src_array[idx],
                self.tgt_array[idx],
                self.src_valid_len[idx]),
                self.label_array[idx])

    def _load_data(self, url: str, md5: str, file_name: str) -> str:
        """
        Downloads the dataset, skipping download if the file 
        already exists and matches the MD5 hash.
        
        Args:
            url (str): URL to download the dataset from.
            md5 (str): MD5 hash to verify the integrity of the downloaded file.
        Returns:
            str: The full content of the dataset.
        """
        tmp_dir = Path("/tmp")
        zip_path = tmp_dir / f"{file_name}.zip"
        extract_dir = tmp_dir / f"{file_name}_extract"
        file_path = Path(self.data_root) / file_name

        if file_path.exists() and self._check_md5(file_path, md5):
            with open(file_path, "r") as f:
                return f.read()

        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        extracted_file = extract_dir / file_name
        if not extracted_file.exists():
            candidates = list(extract_dir.rglob(file_name))
            if candidates:
                extracted_file = candidates[0]
            else:
                raise FileNotFoundError(
                    f"{file_name} not found after extracting {zip_path}"
                )

        if not self._check_md5(extracted_file, md5):
            raise ValueError(f"MD5 mismatch for extracted file {extracted_file}")

        file_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted_file), str(file_path))

        with open(file_path, "r") as f:
            return f.read()

    # Single translate table: normalise unicode whitespace AND insert a
    # space before every punctuation character, all in one C-level pass.
    _CLEAN_TABLE = str.maketrans({
        '\u202f': ' ', '\xa0': ' ', '\u200b': '', '\u2009': ' ',
        ',': ' ,', '.': ' .', '!': ' !', '?': ' ?',
        ';': ' ;', ':': ' :', ')': ' )',
    })
    # Collapse runs of spaces left over from the translate step.
    _MULTI_SPACE_RE = re.compile(r' {2,}')

    def _normalize_sentence(self, sentence: str) -> str:
        """Apply the same normalization used during dataset preprocessing."""
        return self._MULTI_SPACE_RE.sub(
            ' ', sentence.translate(self._CLEAN_TABLE).lower()
        ).strip()

    def _tokenize_sentence(self, sentence: str) -> list[str]:
        """Tokenize a normalized sentence exactly like dataset construction."""
        return sentence.split(' ') + ['<eos>']

    def _preprocess(self, content: str, total_samples: Optional[int] = None) \
        -> tuple[list[str], list[str]]:
        """
        Preprocess the raw content of the dataset.

        Args:
            content (str): Raw content of the dataset file.
            total_samples (Optional[int]): If specified, limits the dataset
                to this many samples.
        Returns:
            tuple[list[str], list[str]]: Lists of source and target sentences.
        """
        # Split lines first so we can truncate *before* doing any
        # per-character work — halves time when total_samples is set.
        lines = content.split('\n')
        if total_samples is not None:
            lines = lines[:total_samples]

        src_sentences: list[str] = []
        tgt_sentences: list[str] = []
        for line in lines:
            parts = line.split('\t', maxsplit=2)
            if len(parts) >= 2:
                s = self._normalize_sentence(parts[0])
                t = self._normalize_sentence(parts[1])
                src_sentences.append(s)
                tgt_sentences.append(t)
        return src_sentences, tgt_sentences
    
    def _build_arrays(self,
                      tokenized_sentences: list[list[str]],
                      vocab: Vocab,
                      is_tgt: bool,
                      return_valid_len: bool = False,
                      ) -> Tensor | Tuple[Tensor, Tensor]:
        """
        Convert tokenized sentences to arrays of token indices.
        Applies padding and truncation as necessary.

        Builds a single flat Python list of indices and materialises one
        tensor — avoids N intermediate tensor allocations + ``torch.stack``.

        Args:
            tokenized_sentences (list[list[str]]): List of tokenized sentences.
            vocab (Vocab): Vocabulary to convert tokens to indices.
            is_tgt (bool): Whether the sentences are target sentences. If True,
                a <bos> token is added at the beginning.
            return_valid_len (bool): If True, also return a tensor of valid
                (non-pad) lengths per sentence.
        Returns:
            Tensor | Tuple[Tensor, Tensor]: Array of shape
                (num_sentences, seq_len) containing token indices, and
                optionally a (num_sentences,) int32 valid-length tensor.
        """
        # Direct dict access bypasses Vocab type-dispatch overhead
        t2i = vocab.token_to_idx
        unk = vocab.unk
        pad = t2i.get('<pad>', unk)
        bos = t2i.get('<bos>', unk) if is_tgt else 0  # unused when not tgt
        seq_len = self.seq_len
        n = len(tokenized_sentences)

        flat: list[int] = []
        valid_lens: list[int] | None = [] if return_valid_len else None

        for sentence in tokenized_sentences:
            if is_tgt:
                row = [bos]
                for tok in sentence:
                    row.append(t2i.get(tok, unk))
            else:
                row = [t2i.get(tok, unk) for tok in sentence]

            rlen = len(row)
            if rlen >= seq_len:
                flat.extend(row[:seq_len])
                if valid_lens is not None:
                    valid_lens.append(seq_len)
            else:
                flat.extend(row)
                flat.extend([pad] * (seq_len - rlen))
                if valid_lens is not None:
                    valid_lens.append(rlen)

        result = torch.tensor(flat, dtype=torch.long).view(n, seq_len)

        if return_valid_len:
            vl = torch.tensor(valid_lens, dtype=torch.int32)
            return result, vl
        return result

    def build(self,
              src_sentences: list[str],
              tgt_sentences: Optional[list[str]] = None,
              ) -> Tuple[Tuple[Tensor, Tensor, Tensor], Tensor]:
        """
        Build a single batch from raw sentence strings, using the
        vocabularies already fitted on the training data.

        Args:
            src_sentences: List of source-language sentences (raw strings).
            tgt_sentences: Optional list of target-language sentences.  If
                ``None``, dummy target / label tensors filled with ``<pad>``
                indices are returned (useful at inference time).

        Returns:
            The same structure as ``__getitem__`` but batched over all
            supplied sentences:
            ``((src_array, tgt_array, src_valid_len), label_array)``
        """
        src_tokenized = [self._tokenize_sentence(self._normalize_sentence(s))
                         for s in src_sentences]
        src_array, src_valid_len = self._build_arrays(
            src_tokenized, self.src_vocab, is_tgt=False,
            return_valid_len=True)

        if tgt_sentences is not None:
            tgt_tokenized = [self._tokenize_sentence(self._normalize_sentence(s))
                             for s in tgt_sentences]
            tgt_full = self._build_arrays(
                tgt_tokenized, self.tgt_vocab, is_tgt=True)
            tgt_array = tgt_full[:, :-1]
            label_array = tgt_full[:, 1:]
        else:
            n = len(src_sentences)
            pad_idx = self.tgt_vocab['<pad>']
            tgt_array = torch.full((n, self.seq_len - 1),
                                   pad_idx, dtype=torch.long)
            # place <bos> at position 0
            tgt_array[:, 0] = self.tgt_vocab['<bos>']
            label_array = torch.full((n, self.seq_len - 1),
                                     pad_idx, dtype=torch.long)

        return (src_array, tgt_array, src_valid_len), label_array

    def _check_md5(self, file_path: str, md5_hash: str) -> bool:
        """Check the MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest() == md5_hash
    

class FraEngDataset(TatoebaDataset):
    """French-English dataset from the Tatoeba project."""
    
    def __init__(self, seq_len: int = 10, token_min_freq: int = 2,
                 data_root: str = './data', total_samples: Optional[int] = None):
        super().__init__(
            seq_len=seq_len,
            token_min_freq=token_min_freq,
            src_lang='en',
            tgt_lang='fr',
            data_url='http://www.manythings.org/anki/fra-eng.zip',
            data_file_name='fra.txt',
            md5_hash='4b438d2eba8bc3afcf68bc5389c721e2',
            data_root=data_root,
            total_samples=total_samples
        )

class GerEngDataset(TatoebaDataset):
    """German-English dataset from the Tatoeba project."""
    
    def __init__(self, seq_len: int = 10, token_min_freq: int = 2,
                 data_root: str = './data', total_samples: Optional[int] = None):
        super().__init__(
            seq_len=seq_len,
            token_min_freq=token_min_freq,
            src_lang='en',
            tgt_lang='de',
            data_url='http://www.manythings.org/anki/deu-eng.zip',
            data_file_name='deu.txt',
            md5_hash='f016e5e7a0de677d7dd0dd5a5424b0b9',
            data_root=data_root,
            total_samples=total_samples
        )
        
    @property
    def test_sentences(self) -> Tuple[list[str], list[str]]:
        """A small set of German-English sentence pairs for testing."""
        pairs = [
            ('go.', 'geh.'),
            ('i lost.', 'ich habe mich verirrt.'),
            ('he\'s calm.', 'er ist ruhig.'),
            ('i\'m home.', 'ich bin zu hause.'),
            ('There is little hope.', 'Es gibt wenig Hoffnung.'),
            ('You should have hope.', 'Du solltest Hoffnung haben.'),
            ('We are all lost.', 'Wir sind alle verloren.'),
            ('I did not mean that.', 'Das habe ich nicht so gemeint.'),
            ('She has a beautiful voice.', 'Sie hat eine schöne Stimme.'),
            ('The weather is nice today.', 'Das Wetter ist heute schön.'),
            ('Do you like reading books?', 'Liest du gerne Bücher?'),
            ('I love programming.', 'Ich liebe Programmierung.'),
            ('I love you.', 'Ich liebe dich.'),
            ('I see a girl with a cat.',
             'Ich sehe ein Mädchen mit einer Katze.'),
            ('Can you help me with my homework?',
             'Kannst du mir bei meinen Hausaufgaben helfen?'),
            ('This is nonsense.', 'Das ist Unsinn.'),
            ('We need to talk.', 'Wir müssen reden.'),
            ('Stay here.', 'Bleib hier.'),
            ('Follow me.', 'Folge mir.'),
            ('I am learning German.', 'Ich lerne Deutsch.'),
            ('What is your name?', 'Wie heißt du?'),
            ('Where is the bathroom?', 'Wo ist die Toilette?'),
            ('How much does this cost?', 'Wie viel kostet das?'),
            ('I would like a coffee.', 'Ich möchte einen Kaffee.'),
            ('Can you help me?', 'Kannst du mir helfen?'),
            ('See you later.', 'Bis später.'),
            ('Good morning.', 'Guten Morgen.'),
            ('My sister reads books in the quiet library.',
            'Meine Schwester liest Bücher in der ruhigen Bibliothek.'),
            ('He forgot his keys on the kitchen table.',
            'Er hat seine Schlüssel auf dem Küchentisch vergessen.'),
            ('Please close the window before you leave.',
            'Bitte schließ das Fenster, bevor du gehst.'),
            ('The train arrives at eight in the morning.',
            'Der Zug kommt um acht Uhr morgens an.'),
            ('We cooked dinner and washed the dishes.',
            'Wir haben Abendessen gekocht und das Geschirr gespült.'),
            ('They walked through the park after school.',
            'Sie gingen nach der Schule durch den Park.'),
            ('He ran out of the door and into the garden.',
            'Er rannte aus der Tür und in den Garten.'),
            ('The quick brown fox jumps over the lazy dog.',
            'Der schnelle braune Fuchs springt über den faulen Hund.'),
            ('Let\'s try a more complex sentence which uses multiple clauses and advanced vocabulary.',
            'Versuchen wir einen komplexeren Satz, der mehrere Klauseln und erweiterten Wortschatz verwendet.'),
            ('Obviously, for less common words, the model struggles.',
            'Offensichtlich hat das Modell bei weniger gebräuchlichen Wörtern Schwierigkeiten.'),
            ('The small boy with the bright blue hat ran quickly down the long, dusty road toward the old wooden house.',
             'Der kleine Junge mit dem hellblauen Hut rannte schnell die lange, staubige Straße hinunter zum alten Holzhaus.'), 
        ]
        
        return [src for src, _ in pairs], [tgt for _, tgt in pairs]
    

def mt_dataloader(
    tatoeba_data: TatoebaDataset, batch_size: int,
    train_ratio: float = 0.8, train: bool = True
) -> DataLoader:
    """Create a DataLoader for the given TatoebaDataset dataset.

    Args:
        tatoeba_data: A TatoebaDataset dataset instance.
        batch_size: Batch size for the DataLoader.
        train_ratio: Ratio of data to use for training.
        train: Whether to return the training DataLoader or validation DataLoader.

    Returns:
        DataLoader yielding (features, labels).
    """
    n_train = int(len(tatoeba_data) * train_ratio)
    indices = torch.randperm(len(tatoeba_data)).tolist()
    if train:
        subset = torch.utils.data.Subset(tatoeba_data, indices[:n_train])
    else:
        subset = torch.utils.data.Subset(tatoeba_data, indices[n_train:])
    return DataLoader(subset, batch_size=batch_size, shuffle=train)

    
def bleu(pred_seq: str, label_seq: str, k: int) -> float:
    """Compute the BLEU."""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, min(k, len_pred) + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


def eval_translations(srcs: list[str],
                      dsts: list[str],
                      preds: list[list[int]],
                      data: TatoebaDataset,
                      bleu_k: int = 2) -> None:
    """Evaluate translations using BLEU score."""
    for en, de, p in zip(srcs, dsts, preds):
        translation = [t for t in data.tgt_vocab.to_tokens(p)]
        if '<eos>' in translation:
            translation = translation[:translation.index('<eos>')]
        translation = [t for t in translation if t != '<pad>']
        pred_text = data._normalize_sentence(' '.join(translation))
        ref_text = data._normalize_sentence(de)
        print(f'{en} => {translation}, bleu: ' +
            f'{bleu(pred_text, ref_text, k=bleu_k):.3f}')
