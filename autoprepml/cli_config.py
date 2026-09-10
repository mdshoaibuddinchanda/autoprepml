"""CLI for managing AutoPrepML configuration and API keys"""

import argparse
import getpass
from .config_manager import AutoPrepMLConfig


def configure_interactive():
    """Interactive configuration wizard"""
    print("\nAutoPrepML Configuration Wizard")
    print("=" * 60)
    print("\nWhich LLM provider would you like to configure?\n")

    providers_list = list(AutoPrepMLConfig.PROVIDERS.keys())
    for idx, provider in enumerate(providers_list, 1):
        info = AutoPrepMLConfig.PROVIDERS[provider]
        print(f"{idx}. {info['name']}")

    print(f"{len(providers_list) + 1}. Skip / Configure later")

    try:
        choice = input(f"\nEnter your choice (1-{len(providers_list) + 1}): ").strip()
        choice_idx = int(choice) - 1

        if choice_idx < 0 or choice_idx >= len(providers_list):
            print("\nConfiguration skipped. You can configure later using 'autoprepml-config'")
            return 0

        provider = providers_list[choice_idx]
        info = AutoPrepMLConfig.PROVIDERS[provider]

        print(f"\nConfiguring {info['name']}")
        print(f"{info['instructions']}\n")

        if provider == "ollama":
            print("Ollama is a local LLM; no API key needed.")
            print("   Install it from https://ollama.ai/ and run: ollama pull llama2")
            return 0

        if api_key := getpass.getpass(
            f"Enter your {info['name']} API key (or press Enter to skip): "
        ).strip():
            AutoPrepMLConfig.set_api_key(provider, api_key)
            print(f"{info['name']} API key saved securely.")
        else:
            print("\nConfiguration skipped for this provider.")
        return 0

    except (ValueError, KeyboardInterrupt):
        print("\n\nConfiguration cancelled.")
        return 2


def main(argv=None):  # sourcery skip: low-code-quality
    """Main CLI entry point for configuration management"""
    parser = argparse.ArgumentParser(
        description="AutoPrepML Configuration - Manage API keys for LLM providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  autoprepml-config                    # Interactive configuration wizard
  autoprepml-config --list             # List all configured API keys
  autoprepml-config --set openai       # Set OpenAI API key
  autoprepml-config --remove anthropic # Remove Anthropic API key
  autoprepml-config --check openai     # Check if OpenAI API key is configured
  autoprepml-config --info             # Show package information

Supported providers: openai, anthropic, google, ollama
        """,
    )

    parser.add_argument("--list", action="store_true", help="List all configured API keys")
    parser.add_argument(
        "--set",
        metavar="PROVIDER",
        help="Set API key for a provider (openai, anthropic, google, ollama)",
    )
    parser.add_argument("--remove", metavar="PROVIDER", help="Remove API key for a provider")
    parser.add_argument(
        "--check", metavar="PROVIDER", help="Check if API key is configured for a provider"
    )
    parser.add_argument(
        "--info", action="store_true", help="Show package and configuration information"
    )

    args = parser.parse_args(argv)

    # If no arguments, run interactive mode
    if not any(vars(args).values()):
        return configure_interactive()

    if args.list:
        AutoPrepMLConfig.list_api_keys()
        return 0

    elif args.set:
        provider = args.set.lower()
        if provider not in AutoPrepMLConfig.PROVIDERS:
            print(f"Unknown provider: {provider}")
            print(f"   Valid providers: {', '.join(AutoPrepMLConfig.PROVIDERS.keys())}")
            return 2

        info = AutoPrepMLConfig.PROVIDERS[provider]
        print(f"\nConfiguring {info['name']}")
        print(f"{info['instructions']}\n")

        if provider == "ollama":
            print("Ollama is a local LLM; no API key needed.")
            print("   Install it from https://ollama.ai/ and run: ollama pull llama2")
            return 0

        if api_key := getpass.getpass(f"Enter your {info['name']} API key: ").strip():
            AutoPrepMLConfig.set_api_key(provider, api_key)
            print(f"{info['name']} API key saved securely.")
        else:
            print("No API key entered. Configuration cancelled.")
        return 0

    elif args.remove:
        provider = args.remove.lower()
        if provider not in AutoPrepMLConfig.PROVIDERS:
            print(f"Unknown provider: {provider}")
            return 2
        AutoPrepMLConfig.remove_api_key(provider)
        return 0

    elif args.check:
        provider = args.check.lower()
        if provider not in AutoPrepMLConfig.PROVIDERS:
            print(f"Unknown provider: {provider}")
            return 2

        api_key = AutoPrepMLConfig.get_api_key(provider)
        info = AutoPrepMLConfig.PROVIDERS[provider]

        if api_key:
            masked = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
            print(f"{info['name']} API key is configured: {masked}")
        elif provider == "ollama":
            print(f"{info['name']} doesn't require an API key (local LLM)")
        else:
            print(f"{info['name']} API key is not configured")
            print(f"   Configure it with: autoprepml-config --set {provider}")
        return 0

    elif args.info:
        try:
            from autoprepml import __version__
        except Exception:
            __version__ = "unknown"

        print("\n" + "=" * 60)
        print("AutoPrepML - AI-Assisted Data Preprocessing")
        print("=" * 60)
        print(f"Version: {__version__}")
        print(f"Config Directory: {AutoPrepMLConfig.CONFIG_DIR}")
        print(f"Config File: {AutoPrepMLConfig.CONFIG_FILE}")
        print("\nSupported LLM Providers:")
        for provider, info in AutoPrepMLConfig.PROVIDERS.items():
            print(f"  {info['name']}")
        print("\nDocumentation: https://github.com/mdshoaibuddinchanda/autoprepml")
        print("=" * 60 + "\n")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
