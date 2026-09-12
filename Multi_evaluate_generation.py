import os
import json
import re
import torch
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from torchvision import transforms

try:
    import spacy
except ImportError:
    spacy = None

class GenerationEvaluator:

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu", dino_model_path="../facebook/dino-vits16"):
        self.device = device
        print(f"🔧 Initializing evaluator on device: {self.device}")

        print("📦 Loading CLIP model...")
        self.clip_model_name = "../laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device)
        self.clip_model.eval()

        print("📦 Loading DINO model...")
        if dino_model_path and os.path.exists(dino_model_path):
            print(f"   Using local DINO model: {dino_model_path}")

            self.dino_processor = AutoImageProcessor.from_pretrained(dino_model_path)
            self.dino_model = AutoModel.from_pretrained(dino_model_path).to(self.device)
            self.use_huggingface_dino = True
        else:
            print("   Loading DINO model with torch.hub...")

            self.dino_model = torch.hub.load("facebookresearch/dino:main", "dino_vits16").to(self.device)

            self.dino_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.use_huggingface_dino = False

        self.dino_model.eval()

        if spacy is not None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("📦 Loading the spaCy English model for relation extraction...")
            except OSError:
                self.nlp = None
                print("⚠️  en_core_web_sm was not detected; relation extraction will fall back to rule-based extraction")
        else:
            self.nlp = None
            print("⚠️  spaCy is not installed; relation extraction will fall back to rule-based extraction")

        self.reference_image_dir = "../data/evaluate_data/reference_images"
        self.project_root = ".."

    def find_image_path(self, image_path):

        candidate1 = os.path.join(self.project_root, image_path)

        candidate2 = os.path.join(self.reference_image_dir, os.path.basename(image_path))

        candidate3 = image_path

        for candidate in [candidate1, candidate2, candidate3]:
            if os.path.exists(candidate):
                return candidate
        return None

    @torch.no_grad()
    def get_clip_image_feature(self, image):

        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        features = self.clip_model.get_image_features(**inputs)
        return features.cpu().numpy()

    @torch.no_grad()
    def get_clip_text_feature(self, text):

        inputs = self.clip_processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        features = self.clip_model.get_text_features(**inputs)
        return features.cpu().numpy()

    @torch.no_grad()
    def get_dino_feature(self, image):

        if self.use_huggingface_dino:

            inputs = self.dino_processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.dino_model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)
        else:

            img_tensor = self.dino_transform(image).unsqueeze(0).to(self.device)
            features = self.dino_model(img_tensor)

        return features.cpu().numpy()

    def calculate_cosine_similarity(self, feat1, feat2):

        return cosine_similarity(feat1, feat2)[0][0]

    def extract_relation_text(self, prompt):

        if not prompt or not isinstance(prompt, str):
            return ""

        text = prompt.strip()
        if not text:
            return ""

        if self.nlp is not None:
            try:
                doc = self.nlp(text)
                root_candidates = []
                for token in doc:
                    if token.pos_ in {"VERB", "AUX"} and token.lemma_.lower() not in {"be", "am", "is", "are", "was", "were"}:
                        root_candidates.append(token)

                if root_candidates:
                    root = root_candidates[0]
                    phrase_tokens = [root.lemma_.lower()]
                    for child in root.children:
                        if child.dep_ in {"prep", "prt", "advmod", "xcomp", "acomp", "aux", "neg"}:
                            if child.text.lower() not in {"a", "an", "the", "their", "them"}:
                                phrase_tokens.append(child.text.lower())
                    cleaned = []
                    for token in phrase_tokens:
                        token = token.strip().lower()
                        if token and token not in {"a", "an", "the", "their", "them"}:
                            cleaned.append(token)
                    if cleaned:
                        return " ".join(cleaned)
            except Exception:
                pass

        patterns = [
            r"\b(?:is|are|was|were|be|being)\s+([a-z]+(?:\s+(?:next|beside|behind|opposite|under|over|on|in|with|between|against|around|inside|outside|to|facing|alongside|against|while|across))*)",
            r"\b([a-z]+(?:\s+(?:next|beside|behind|opposite|under|over|on|in|with|between|against|around|inside|outside|to|facing|alongside|across|while))*)",
        ]

        normalized = text.lower()
        for pat in patterns:
            match = re.search(pat, normalized, flags=re.IGNORECASE)
            if match:
                candidate = match.group(1).strip().lower()
                if candidate and candidate not in {"a", "an", "the"}:
                    return candidate

        candidate_match = re.search(r"\b([a-z]+(?:\s+(?:next|beside|behind|opposite|under|over|on|in|with|between|against|around|inside|outside|to|facing|alongside|across|while))+)\b", normalized)
        if candidate_match:
            return candidate_match.group(1).strip()

        return ""

    def evaluate_sample(self, generated_image_path, metadata):

        gen_image = Image.open(generated_image_path).convert("RGB")

        if "subjects" in metadata:
            subject_paths = [subj["file"] for subj in metadata["subjects"]]
        else:
            subject_paths = metadata.get("subject_files", [])

        reference_images = []
        for img_path in subject_paths:
            full_path = self.find_image_path(img_path)
            if not full_path:
                print(f"⚠️  Reference image not found: {img_path}")
                return None
            ref_image = Image.open(full_path).convert("RGB")
            reference_images.append(ref_image)

        gen_clip_feat = self.get_clip_image_feature(gen_image)
        gen_dino_feat = self.get_dino_feature(gen_image)

        clip_i_scores = []
        dino_scores = []
        for ref_image in reference_images:
            ref_clip_feat = self.get_clip_image_feature(ref_image)
            ref_dino_feat = self.get_dino_feature(ref_image)

            clip_i = self.calculate_cosine_similarity(gen_clip_feat, ref_clip_feat)
            dino = self.calculate_cosine_similarity(gen_dino_feat, ref_dino_feat)

            clip_i_scores.append(clip_i)
            dino_scores.append(dino)

        avg_clip_i = np.mean(clip_i_scores)
        avg_dino = np.mean(dino_scores)

        m_dino = float(np.prod(dino_scores))

        prompt = metadata.get("prompt") or metadata.get("filled_prompt")
        if not prompt:
            print(f"⚠️  Metadata has no prompt or filled_prompt field; skipping this sample")
            return None

        text_feat = self.get_clip_text_feature(prompt)
        clip_t = self.calculate_cosine_similarity(gen_clip_feat, text_feat)

        relation_text = self.extract_relation_text(prompt)
        relation_feat = self.get_clip_text_feature(relation_text) if relation_text else None
        clip_r = self.calculate_cosine_similarity(gen_clip_feat, relation_feat) if relation_feat is not None else None

        return {
            "sample_id": os.path.basename(generated_image_path).replace(".jpg", ""),
            "prompt": prompt,
            "relation_text": relation_text,
            "subjects": subject_paths,
            "clip_i_scores": [float(s) for s in clip_i_scores],
            "avg_clip_i": float(avg_clip_i),
            "dino_scores": [float(s) for s in dino_scores],
            "avg_dino": float(avg_dino),
            "m_dino": m_dino,
            "clip_t": float(clip_t),
            "clip_r": float(clip_r) if clip_r is not None else None
        }

    def evaluate_directory(self, eval_dir, output_path=None):

        eval_dir = Path(eval_dir)
        print(f"\n🚀 Starting evaluation of directory: {eval_dir}")

        image_files = list(eval_dir.glob("*.jpg"))

        image_files = [f for f in image_files if "_" in f.stem and f.stem.split("_")[0].isdigit() and f.stem.split("_")[1].isdigit()]

        image_files.sort(key=lambda x: (int(x.stem.split("_")[0]), int(x.stem.split("_")[1])))

        print(f"📸 Found {len(image_files)} generated samples")

        results = []
        failed = []

        for img_path in tqdm(image_files, desc="Evaluation progress"):

            metadata_path = img_path.with_name(f"{img_path.stem}_metadata.json")
            if not metadata_path.exists():
                print(f"⚠️  Metadata file not found: {metadata_path}")
                failed.append(str(img_path))
                continue

            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            try:
                sample_result = self.evaluate_sample(str(img_path), metadata)
                if sample_result:
                    results.append(sample_result)
            except Exception as e:
                print(f"❌ Evaluation of sample {img_path} failed: {str(e)}")
                failed.append(str(img_path))

        all_clip_i = [r["avg_clip_i"] for r in results]
        all_dino = [r["avg_dino"] for r in results]
        all_m_dino = [r["m_dino"] for r in results]
        all_clip_t = [r["clip_t"] for r in results]
        all_clip_r = [r["clip_r"] for r in results if r.get("clip_r") is not None]

        stats = {
            "total_samples": len(results),
            "failed_samples": len(failed),
            "metrics": {
                "avg_clip_i": float(np.mean(all_clip_i)),
                "std_clip_i": float(np.std(all_clip_i)),
                "avg_dino": float(np.mean(all_dino)),
                "std_dino": float(np.std(all_dino)),
                "avg_m_dino": float(np.mean(all_m_dino)),
                "std_m_dino": float(np.std(all_m_dino)),
                "avg_clip_t": float(np.mean(all_clip_t)),
                "std_clip_t": float(np.std(all_clip_t)),
                "avg_clip_r": float(np.mean(all_clip_r)) if all_clip_r else None,
                "std_clip_r": float(np.std(all_clip_r)) if all_clip_r else None
            },
            "failed_samples": failed,
            "detailed_results": results
        }

        print("\n" + "="*50)
        print("📊 Evaluation statistics:")
        print(f"   Total samples: {stats['total_samples']}")
        print(f"   Failed samples: {stats['failed_samples']}")
        print(f"\n   🔍 Identity-preservation metrics:")
        print(f"     Average CLIP-I similarity: {stats['metrics']['avg_clip_i']:.4f} (±{stats['metrics']['std_clip_i']:.4f})")
        print(f"     Average DINO similarity: {stats['metrics']['avg_dino']:.4f} (±{stats['metrics']['std_dino']:.4f})")
        print(f"     Average M-DINO (multi-subject product): {stats['metrics']['avg_m_dino']:.4f} (±{stats['metrics']['std_m_dino']:.4f})")
        print(f"\n   📝 Text-alignment metrics:")
        print(f"     Average CLIP-T similarity: {stats['metrics']['avg_clip_t']:.4f} (±{stats['metrics']['std_clip_t']:.4f})")
        if stats['metrics']['avg_clip_r'] is not None:
            print(f"     Average CLIP-R similarity: {stats['metrics']['avg_clip_r']:.4f} (±{stats['metrics']['std_clip_r']:.4f})")
        else:
            print("     Average CLIP-R similarity: not calculated (no valid relation phrase was extracted)")
        print("="*50)

        if output_path is None:
            output_path = eval_dir / "evaluation_results.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"\n💾 Detailed evaluation results saved to: {output_path}")

        return stats

if __name__ == "__main__":
    evaluator = GenerationEvaluator()

    eval_directory = "../inference_result"
    output_file = "../inference_result/evaluation_results.json"

    results = evaluator.evaluate_directory(eval_directory, output_file)
