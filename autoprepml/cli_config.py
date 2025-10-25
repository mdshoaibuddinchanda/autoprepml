"""CLI for managing AutoPrepML configuration and API keys"""
import sys
import argparse
import getpass
from .config_manager import AutoPrepMLConfig


def configure_interactive():
    """Interactive configuration wizard"""
    print("\n🎯 AutoPrepML Configuration Wizard")
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
            print("\n✅ Configuration skipped. You can configure later using 'autoprepml-config'")
            return

        provider = providers_list[choice_idx]
        info = AutoPrepMLConfig.PROVIDERS[provider]

        print(f"\n📝 Configuring {info['name']}")
        print(f"ℹ️  {info['instructions']}\n")

        if provider == 'ollama':
            print("✅ Ollama is a local LLM - no API key needed!")
            print("   Install it from https://ollama.ai/ and run: ollama pull llama2")
            return

        if api_key := getpass.getpass(
            f"Enter your {info['name']} API key (or press Enter to skip): "
        ).strip():
            AutoPrepMLConfig.set_api_key(provider, api_key)
            print(f"✅ {info['name']} API key saved securely!")
        else:
            print("\n✅ Configuration skipped for this provider.")

    except (ValueError, KeyboardInterrupt):
        print("\n\n✅ Configuration cancelled.")


def main():    # sourcery skip: low-code-quality
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
        """
    )

    parser.add_argument('--list', action='store_true', 
                       help='List all configured API keys')
    parser.add_argument('--set', metavar='PROVIDER', 
                       help='Set API key for a provider (openai, anthropic, google, ollama)')
    parser.add_argument('--remove', metavar='PROVIDER', 
                       help='Remove API key for a provider')
    parser.add_argument('--check', metavar='PROVIDER', 
                       help='Check if API key is configured for a provider')
    parser.add_argument('--info', action='store_true',
                       help='Show package and configuration information')

    args = parser.parse_args()

    # If no arguments, run interactive mode
    if not any(vars(args).values()):
        configure_interactive()
        return

    if args.list:
        AutoPrepMLConfig.list_api_keys()

    elif args.set:
        provider = args.set.lower()
        if provider not in AutoPrepMLConfig.PROVIDERS:
            print(f"❌ Unknown provider: {provider}")
            print(f"   Valid providers: {', '.join(AutoPrepMLConfig.PROVIDERS.keys())}")
            sys.exit(1)

        info = AutoPrepMLConfig.PROVIDERS[provider]
        print(f"\n📝 Configuring {info['name']}")
        print(f"ℹ️  {info['instructions']}\n")

        if provider == 'ollama':
            print("✅ Ollama is a local LLM - no API key needed!")
            print("   Install it from https://ollama.ai/ and run: ollama pull llama2")
            return

        if api_key := getpass.getpass(
            f"Enter your {info['name']} API key: "
        ).strip():
            AutoPrepMLConfig.set_api_key(provider, api_key)
            print(f"✅ {info['name']} API key saved securely!")
        else:
            print("❌ No API key entered. Configuration cancelled.")

    elif args.remove:
        provider = args.remove.lower()
        if provider not in AutoPrepMLConfig.PROVIDERS:
            print(f"❌ Unknown provider: {provider}")
            sys.exit(1)
        AutoPrepMLConfig.remove_api_key(provider)

    elif args.check:
        provider = args.check.lower()
        if provider not in AutoPrepMLConfig.PROVIDERS:
            print(f"❌ Unknown provider: {provider}")
            sys.exit(1)

        api_key = AutoPrepMLConfig.get_api_key(provider)
        info = AutoPrepMLConfig.PROVIDERS[provider]

        if api_key:
            masked = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
            print(f"✅ {info['name']} API key is configured: {masked}")
        elif provider == 'ollama':
            print(f"ℹ️  {info['name']} doesn't require an API key (local LLM)")
        else:
            print(f"❌ {info['name']} API key is not configured")
            print(f"   Configure it with: autoprepml-config --set {provider}")

    elif args.info:
        try:
            from autoprepml import __version__
        except Exception:
            __version__ = "unknown"

        print("\n" + "=" * 60)
        print("🤖 AutoPrepML - AI-Assisted Data Preprocessing")
        print("=" * 60)
        print(f"Version: {__version__}")
        print(f"Config Directory: {AutoPrepMLConfig.CONFIG_DIR}")
        print(f"Config File: {AutoPrepMLConfig.CONFIG_FILE}")
        print("\nSupported LLM Providers:")
        for provider, info in AutoPrepMLConfig.PROVIDERS.items():
            print(f"  • {info['name']}")
        print("\nDocumentation: https://github.com/mdshoaibuddinchanda/autoprepml")
        print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
