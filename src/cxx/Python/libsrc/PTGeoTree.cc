

#include <VecGeom/base/Transformation3D.h>
#include "PTGeoTree.hh"

using namespace std;

std::vector<std::shared_ptr<Prompt::GeoTree::Node>> Prompt::GeoTree::Node::allNodes = std::vector<std::shared_ptr<Prompt::GeoTree::Node>>{};

Prompt::GeoTree::GeoTree()
:m_root(std::make_shared<GeoTree::Node>())
{
}

Prompt::GeoTree::~GeoTree() {}

shared_ptr<Prompt::GeoTree::Node> Prompt::GeoTree::getRoot() { return m_root; }

vector<shared_ptr<Prompt::GeoTree::Node>> Prompt::GeoTree::findMotherNodeByPhysical(int num)
{
  auto motherNodes = vector<shared_ptr<GeoTree::Node>>();
  for(auto node : m_root->allNodes)
  {
    if(std::find(node->childPhysicalID.begin(), node->childPhysicalID.end(), num) != node->childPhysicalID.end())
      motherNodes.push_back(node);
  }
  return motherNodes;
}

shared_ptr<Prompt::GeoTree::Node> Prompt::GeoTree::findNodeByPhysical(int num)
{
  return findNodeByPhysical(m_root, num);
}

shared_ptr<Prompt::GeoTree::Node> Prompt::GeoTree::findNodeByPhysical( const shared_ptr<Prompt::GeoTree::Node> &node, int num)
{
  if(!node)
    return nullptr;
  if (num == node->physical)
    return node;
  for (auto childptr : node->child)
	{
		return findNodeByPhysical(childptr, num);
	}
  	return nullptr;
}

std::vector<std::shared_ptr<Prompt::GeoTree::Node>> Prompt::GeoTree::findNodeByLogical(int num)
{
  std::vector<std::shared_ptr<Node>> logicalnode;
  findNodeByLogical(m_root, num, logicalnode);
  return logicalnode;
}


void Prompt::GeoTree::findNodeByLogical(const std::shared_ptr<Prompt::GeoTree::Node> &node, int num, std::vector<std::shared_ptr<Node>>& logicalnode)
{
  if(!node)
    return;
  if (num == node->logical)
  {
    logicalnode.push_back(node);
  }
  for (auto childptr : node->child)
  {
    findNodeByLogical(childptr, num, logicalnode);
  }
}

void Prompt::GeoTree::print(const std::shared_ptr<Prompt::GeoTree::Node> &node, int layer, std::vector<std::vector<int>> &printArray, bool phys)
{
  try
  {
    if(phys)
      printArray.at(layer).push_back(node->physical);
    else
    {
      for(auto v: node->childPhysicalID)
        printArray.at(layer).push_back(v);
    }
  }
  catch (...)
  {
    printArray.push_back(vector<int>());
    if(phys)
      printArray.at(layer).push_back(node->physical);
    else
    {
      for(auto v: node->childPhysicalID)
        printArray.at(layer).push_back(v);
    }
  }
  for(auto pointptr: node->child)
  {
    print(pointptr,layer+1, printArray, phys);
  }
}

void Prompt::GeoTree::print(bool phys)
{
  vector<vector<int>> printArray;
  print(m_root, 0, printArray, phys);
  for(vector<int> varvector : printArray)
  {
    for(int var : varvector)
      cout << var << " ";
    cout << "\n";
  }
}
