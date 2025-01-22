# API Reference

This page contains the API reference for the Power Attention library.

## High-Level Interface

The main entry point for using symmetric power attention in your models.

Main implementation of symmetric power attention, which generalizes linear transformers using symmetric power embeddings. 
Provides O(n) complexity for long sequences through an efficient RNN formulation.

::: power_attention.power_full.power_full
    options:
        show_root_heading: false
        show_source: true
        heading_level: 3

## Low-Level Components

These components implement the linear transformer with symmetric power embeddings. They're exposed for advanced usage
and custom implementations.

### Core Attention

Computes attention scores using symmetric power embeddings, equivalent to raising attention weights to an even power.

::: power_attention._attention
    options:
        show_root_heading: true
        show_source: true
        members: [attention]
        heading_level: 4

### State Management

Functions for managing the RNN state representation, which achieves massive memory savings through symmetric tensors.

#### State Expansion

Computes expanded state vectors using symmetric power embeddings, achieving up to 96% memory reduction for deg=4.

::: power_attention._update_state
    options:
        show_root_heading: true
        show_source: true
        members: [update_state]
        heading_level: 5

#### Query-State Interaction

Computes how queries interact with the compressed state representation from previous chunks.

::: power_attention._query_state
    options:
        show_root_heading: true
        show_source: true
        members: [query_state]
        heading_level: 5

### Recurrent Processing

Implements the RNN state update equations for O(n) processing of long sequences.

::: power_attention._discumsum
    options:
        show_root_heading: true
        show_source: true
        members: [discumsum]
        heading_level: 4

## Utility Functions

Helper functions for testing and benchmarking.

Create sample inputs for testing power attention, with appropriate initialization for stable training.

::: power_attention.power_full
    options:
        show_root_heading: true
        show_source: true
        members: [create_inputs]
        heading_level: 3 