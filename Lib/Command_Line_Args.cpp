#include "Command_Line_Args.h"
#include "Utils.h"

#include <string> 
#include <memory>
#include <fstream> 
#include <iostream>
#include <filesystem> 

std::string Command_Line_Args::get_usage()
{
    std::ostringstream usage;

    usage << "Allowed arguments:\n"
          << "  --alpha   : Angle to partition the 2-D plane\n"
          << "  --rho     : Number of different DFS orderings for exploitation\n"
          << "  --topk    : DFS candidates to polish with 2-opt (default 2)\n"
          << "  --ls      : Local search level: 0=2-opt, 1=+Or-opt/relocate, 2=+3-opt (default 0)\n"
          << "  --input   : Input file path\n"
          << " --output   : Output file path\n\n"

          << "Mandatory arguments:\n"
          << "  alpha and rho\n\n"

          << "Optional arguments:\n"
          << "  topk, ls, input and output\n\n"

          << "Acceptable usage examples:\n"
          << "  ./executable --alpha=<alpha> --rho=<rho>\n"
          << "  ./executable --alpha=<alpha> --rho=<rho> --topk=<k>\n"
          << "  ./executable --alpha=<alpha> --rho=<rho> --ls=2\n"
          << "  ./executable --alpha=<alpha> --rho=<rho> --output=<output>\n"
          << "  ./executable --alpha=<alpha> --rho=<rho> --input=<input>\n"
          << "  ./executable --alpha=<alpha> --rho=<rho> --input=<input> --output=<output>\n";

    return usage.str();
}

void Command_Line_Args::set_alpha(int argc, char** argv)
{
    bool alpha_passed = false;

    for(int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg.rfind("--alpha=", 0) == 0) 
        {
            if (alpha_passed)
            {
                HANDLE_ERROR("Alpha must be passed only once in command line arguments\n" + this->get_usage(), true);
            }
            alpha_passed = true;
            this->alpha = std::stod(arg.substr(8));
            if (this->alpha <= 0 || this->alpha > 360)
            {
                HANDLE_ERROR("Alpha must be in the range (0, 360]", true);
            }
        } 
    }
    
    if(!alpha_passed)
    {
        HANDLE_ERROR("Alpha is not passed\n" + this->get_usage(), true);
    }

    return;
}

void Command_Line_Args::set_rho(int argc, char** argv)
{
    bool rho_passed = false;

    for(int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg.rfind("--rho=", 0) == 0) 
        {
            if (rho_passed)
            {
                HANDLE_ERROR("Rho must be passed exactly once in command line arguments\n" + this->get_usage(), true);
            }
            rho_passed = true;
            this->rho = std::stoi(arg.substr(6));
            if (this->rho <= 0)
            {
                HANDLE_ERROR("Rho must be positive", true);
            }
        } 
    }

    if(!rho_passed)
    {
        HANDLE_ERROR("Rho is not passed\n" + this->get_usage(), true);
    }

    return;
}

void Command_Line_Args::set_input_stream(int argc, char** argv)
{
    input_stream = &std::cin;
    bool input_file_passed = false;

    for (int i = 1; i < argc; i++) 
    {
        std::string arg = argv[i];
        if (arg.rfind("--input=", 0) == 0) 
        {
            if(input_file_passed)
            {
                HANDLE_ERROR("Input file path must be passed at most once\n" + this->get_usage(), true);
            }
            input_file_passed = true;
            std::string filename = arg.substr(8);
            input_file_owner = std::make_unique<std::ifstream>(filename);
            if (!input_file_owner->is_open())
            {
                HANDLE_ERROR("Cannot open input file: " + filename, true);
            }
            input_stream = input_file_owner.get();
        } 
    }
}

void Command_Line_Args::set_output_stream(int argc, char** argv)
{
    output_stream = &std::cout;
    bool output_file_passed = false;
    for (int i = 1; i < argc; i++) 
    {
        std::string arg = argv[i];
        if (arg.rfind("--output=", 0) == 0) 
        {
            if(output_file_passed)
            {
                HANDLE_ERROR("Output file path must be passed at most once\n" + this->get_usage(), true);
            }
            output_file_passed = true;
            std::string filename = arg.substr(9);
            std::filesystem::path out_path(filename);

            if (out_path.has_parent_path()) 
            {
                std::filesystem::create_directories(out_path.parent_path());
            }
            
            output_file_owner = std::make_unique<std::ofstream>(out_path, std::ios::trunc);
            if (!output_file_owner->is_open()) 
            {
                HANDLE_ERROR("Cannot open output file: " + filename, true);
            }
            output_stream = output_file_owner.get();
        } 
    }
}

void Command_Line_Args::set_topk(int argc, char** argv)
{
    this->topk = 2;  // default: 127/127 time vs BP-MDS; raise for gap at cost of time
    bool topk_passed = false;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg.rfind("--topk=", 0) == 0)
        {
            if (topk_passed)
            {
                HANDLE_ERROR("topk must be passed at most once\n" + this->get_usage(), true);
            }
            topk_passed = true;
            this->topk = std::stoi(arg.substr(7));
            if (this->topk <= 0)
            {
                HANDLE_ERROR("topk must be positive", true);
            }
        }
    }
}

void Command_Line_Args::set_ls(int argc, char** argv)
{
    this->ls = 0;  // default: 2-opt only (matched R006–R010 time)
    bool ls_passed = false;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg.rfind("--ls=", 0) == 0)
        {
            if (ls_passed)
            {
                HANDLE_ERROR("ls must be passed at most once\n" + this->get_usage(), true);
            }
            ls_passed = true;
            this->ls = std::stoi(arg.substr(5));
            if (this->ls < 0 || this->ls > 2)
            {
                HANDLE_ERROR("ls must be 0, 1, or 2", true);
            }
        }
    }
}

Command_Line_Args::Command_Line_Args(int argc, char** argv) 
{
    this->set_alpha(argc, argv);
    this->set_rho(argc, argv);
    this->set_topk(argc, argv);
    this->set_ls(argc, argv);

    this->set_input_stream(argc, argv);
    this->set_output_stream(argc, argv);
}

std::istream& Command_Line_Args::input() const 
{ 
    

    return *input_stream; 
}

std::ostream& Command_Line_Args::output() const
{ 
    

    return *output_stream; 
}

double Command_Line_Args::get_alpha() const 
{ 
    

    return alpha; 
}

int Command_Line_Args::get_rho() const 
{ 
    

    return rho; 
}

int Command_Line_Args::get_topk() const
{
    return topk;
}

int Command_Line_Args::get_ls() const
{
    return ls;
}