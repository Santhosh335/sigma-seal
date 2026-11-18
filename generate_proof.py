"""
Month 3: generate and verify a zero-knowledge proof that
sigma_seal_model.onnx, run on sample_input.npy, produced a specific,
provable output -- WITHOUT the verifier needing the model weights or
re-running anything themselves.
"""
import asyncio
import json

import ezkl
import numpy as np


async def main():
    model_path = "sigma_seal_model.onnx"
    settings_path = "settings.json"
    compiled_path = "model.compiled"
    pk_path = "proving.key"
    vk_path = "verifying.key"
    input_path = "input.json"
    proof_path = "proof.pf"
    witness_path = "witness.json"

    sample_input = np.load("sample_input.npy")
    json.dump({"input_data": [sample_input.reshape(-1).tolist()]}, open(input_path, "w"))

    print("Step 1/7: generating settings...")
    assert ezkl.gen_settings(model_path, settings_path)

    print("Step 2/7: calibrating settings (checks quantization accuracy loss)...")
    assert ezkl.calibrate_settings(input_path, model_path, settings_path, "resources")

    print("Step 3/7: compiling circuit...")
    assert ezkl.compile_circuit(model_path, compiled_path, settings_path)

    print("Step 4/7: downloading SRS files (one-time, may take a while)...")
    await ezkl.get_srs(settings_path)

    print("Step 5/7: setup -- generating proving and verifying keys...")
    assert ezkl.setup(compiled_path, vk_path, pk_path)
    print("  -> verifying.key is the file you'd publish/share for others to verify proofs")

    print("Step 6/7: generating witness...")
    ezkl.gen_witness(input_path, compiled_path, witness_path)

    print("Step 7/7: generating proof...")
    proof = ezkl.prove(witness_path, compiled_path, pk_path, proof_path)
    print(f"  Proof saved to {proof_path}")

    print("\nVerifying the proof (this is what anyone else can do, with just")
    print("verifying.key, settings.json, and proof.pf -- no model weights needed)...")
    result = ezkl.verify(proof_path, settings_path, vk_path)
    print(f"\nVERIFICATION RESULT: {result}")
    if result:
        print("SUCCESS: proof verified. This cryptographically confirms the model")
        print("produced this exact output on this exact input, without the verifier")
        print("needing the model weights or re-running anything.")


if __name__ == "__main__":
    asyncio.run(main())