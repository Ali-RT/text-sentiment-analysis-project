"""Console script for text_sentiment_analysis."""
import click

from text_sentiment_analysis.main import train_sentiment_network


@click.group()
def cli():
    pass


@cli.command()
def train_model():
    """
    Train a new model.
    """
    click.echo(f"Training model on reviews")
    train_sentiment_network()


if __name__ == "__main__":
    train_model()
