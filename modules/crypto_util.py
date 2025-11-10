"""
Cryptographic utilities for message signing and verification using ECDSA.
Implements asymmetric encryption for watermark authentication.
"""
import os
from ecdsa import SigningKey, VerifyingKey, NIST256p
from ecdsa.util import sigencode_string, sigdecode_string


def generate_keys(sk_path="sk.pem", pk_path="pk.pem"):
    """
    Generate and save ECDSA key pair using NIST256p curve.
    
    Args:
        sk_path (str): Path to save private key
        pk_path (str): Path to save public key
    
    Returns:
        tuple: (sk_path, pk_path) paths to saved keys
    """
    # Generate private key
    sk = SigningKey.generate(curve=NIST256p)
    
    # Get public key
    pk = sk.get_verifying_key()
    
    # Save private key
    with open(sk_path, 'wb') as f:
        f.write(sk.to_pem())
    
    # Save public key
    with open(pk_path, 'wb') as f:
        f.write(pk.to_pem())
    
    print(f"Keys generated and saved: {sk_path}, {pk_path}")
    return sk_path, pk_path


def sign_message(sk_path, message_str):
    """
    Sign a message using private key.
    
    Args:
        sk_path (str): Path to private key file
        message_str (str): Message to sign (will be UTF-8 encoded)
    
    Returns:
        bytes: Signature bytes
    """
    # Load private key
    with open(sk_path, 'rb') as f:
        sk = SigningKey.from_pem(f.read())
    
    # Sign message (UTF-8 encoded)
    message_bytes = message_str.encode('utf-8')
    signature = sk.sign(message_bytes, sigencode=sigencode_string)
    
    return signature


def verify_signature(pk_path, signature_bytes, message_str):
    """
    Verify a message signature using public key.
    
    Args:
        pk_path (str): Path to public key file
        signature_bytes (bytes): Signature to verify
        message_str (str): Original message (will be UTF-8 encoded)
    
    Returns:
        bool: True if signature is valid, False otherwise
    """
    try:
        # Load public key
        with open(pk_path, 'rb') as f:
            pk = VerifyingKey.from_pem(f.read())
        
        # Verify signature
        message_bytes = message_str.encode('utf-8')
        pk.verify(signature_bytes, message_bytes, sigdecode=sigdecode_string)
        return True
    except Exception as e:
        print(f"Signature verification failed: {e}")
        return False
